package controllers

import models._
import models.daos._
import org.apache.pekko.stream.scaladsl.{FileIO, Source}

import javax.inject._
import play.api._
import play.api.http.HttpEntity

import java.nio.file.{Files, Paths}
import play.api.libs.json._
import play.api.mvc._
import play.api.libs.ws._
import play.api.mvc.MultipartFormData.{DataPart, FilePart}

import java.io.File
import scala.concurrent.duration.DurationInt
import scala.concurrent.{ExecutionContext, Future}
import scala.language.postfixOps

@Singleton
class HomeController @Inject()(
    cc: ControllerComponents,
    config: Configuration,
    ws: WSClient,
    chatDAO: ChatDAO,
    feedbackDAO: FeedbackDAO
) (implicit ec: ExecutionContext) extends AbstractController(cc) {

  def index() = Action.async { implicit request: Request[AnyContent] =>
    chatDAO.getLastN(20).map { chats =>
      Ok(views.html.index(config, chats, None))
    }
  }

  def add() = Action.async { implicit request: Request[AnyContent] =>
    ws
      .url(s"${config.get[String]("server_url")}/get_documents")
      .withRequestTimeout(5 minutes)
      .get()
      .map(files => {
        Ok(views.html.add(files.json.as[Seq[JsObject]]))
      })
  }

  def search() = Action.async { implicit request: Request[AnyContent] =>
    val json = request.body.asJson.getOrElse(Json.obj()).as[JsObject]
    val query = (json \ "query").as[String]
    val history = (json \ "history").as[Seq[JsObject]]
    val docs = (json \ "docs").as[Seq[JsObject]]
    val chatId = (json \ "id").as[String]
    val messageOffset = (json \ "message_offset").as[Int]
    val askTime = System.currentTimeMillis()

    ws
      .url(s"${config.get[String]("server_url")}/chat")
      .withRequestTimeout(5 minutes)
      .post(Json.obj(
        "prompt" -> query,
        "history" -> history,
        "docs" -> docs
      ))
      .flatMap(response =>
        // Check if the chat exists
        chatDAO.get(chatId).flatMap(_ match {
          case Some(chat) => Future.successful(chat)
          case None => {
            // We need to create the chat but before doing so, we need a title that summarizes this
            ws
              .url(s"${config.get[String]("server_url")}/create_title")
              .withRequestTimeout(5 minutes)
              .post(Json.obj(
                "question" -> query
              ))
              .flatMap(titleResponse => {
                val title = (titleResponse.json.as[JsObject] \ "title").as[String]
                chatDAO.add(Chat(
                  chatId, title, System.currentTimeMillis()
                ))
              })
          }
        }).flatMap { chat =>
          // Now insert the chat messages
          val history = (response.json \ "history").as[Seq[JsObject]].drop(messageOffset - 1)
          val (systemMessages, historyMessages) = history.partition(message => (message \ "role").as[String] == "system")

          val chatMessages = (systemMessages.headOption match {
            case Some(message) => Seq(ChatMessage(chat.id, messageOffset, askTime - 1, (message \ "content").as[String], "system", Json.stringify(Json.arr()), None, false))
            case None => Nil
          }) ++ historyMessages.map {message =>
            if ((message \ "role").as[String] == "user")
              ChatMessage(chat.id, (if (systemMessages.size > 0) 1 else 0) + messageOffset, askTime, (message \ "content").as[String], "user", Json.stringify(Json.arr()), None, false)
            else
              ChatMessage(chat.id, (if (systemMessages.size > 0) 1 else 0) + messageOffset + 1, System.currentTimeMillis(), (message \ "content").as[String], "assistant",
                Json.stringify((response.json.as[JsObject] \ "documents").as[JsValue]), (response.json.as[JsObject] \ "rewritten").asOpt[String], (response.json.as[JsObject] \ "fetched_new_documents").as[Boolean])
          }
          Future.sequence(chatMessages.map(message => chatDAO.addChatMessage(message))).map {_ =>
            Ok(response.json)
          }
        }
      )
  }

  def download(file: String) = Action.async { implicit request: Request[AnyContent] =>
    ws.url(s"${config.get[String]("server_url")}/get_document")
      .withRequestTimeout(5.minutes)
      .post(Json.obj("filename" -> file))
      .map { response =>
        if (response.status == 200) {
          // Get the content type and filename from headers
          val contentType = response.header("Content-Type").getOrElse("application/octet-stream")
          val disposition = response.header("Content-Disposition").getOrElse("")
          val filenameRegex = """filename="?(.+)"?""".r
          val downloadFilename = filenameRegex.findFirstMatchIn(disposition).map(_.group(1)).getOrElse(file)

          // Stream the response body to the user
          Result(
            header = ResponseHeader(200, Map(
              "Content-Disposition" -> s"""attachment; filename="$downloadFilename"""",
              "Content-Type" -> contentType
            )),
            body = HttpEntity.Streamed(
              response.bodyAsSource,
              response.header("Content-Length").map(_.toLong),
              Some(contentType)
            )
          )
        } else {
          // Handle error cases
          Status(response.status)(s"Error: ${response.statusText}")
        }
      }
  }

  def upload = Action.async(parse.multipartFormData) { implicit request =>
    request.body.file("file").map { file =>
    // Copy over file
    val filename = Paths.get(file.filename).getFileName
    val dataFolder = config.get[String]("data_folder")
    val filePath = new java.io.File(s"$dataFolder/$filename")

    // Create folder if it doesn't exist yet
    val dataFolderFile = new File(dataFolder)
    if (!dataFolderFile.exists()) {
      if (dataFolderFile.mkdirs()) {} else {
        throw new RuntimeException(s"Failed to create directory $dataFolder.")
      }
    } else if (!dataFolderFile.isDirectory) {
      throw new RuntimeException(s"$dataFolder exists but is not a directory.")
    }

    file.ref.copyTo(filePath)

    // Prepare the file as a FilePart
    val filePart = FilePart(
      key = "file",
      filename = filePath.getName,
      contentType = Some(Files.probeContentType(filePath.toPath)),
      ref = FileIO.fromPath(filePath.toPath)
    )

    // Send the file as multipart/form-data
    ws.url(s"${config.get[String]("server_url")}/add_document")
      .withRequestTimeout(5.minutes)
      .post(Source(List(filePart)))
      .map { response =>
        // Remove the file locally
        filePath.delete()

        response.status match {
          case OK =>
            Redirect(routes.HomeController.add()).flashing("success" -> "Added file to the database.")
          case _ => Redirect(routes.HomeController.add()).flashing("error" -> "Adding file to database failed.")
        }
      }.recover {
        case e: Exception => {
          filePath.delete()
          Redirect(routes.HomeController.add()).flashing("error" -> s"Internal server error: ${e.getMessage()}")
        }
      }
    }.getOrElse {
      Future.successful(Redirect(routes.HomeController.add()).flashing(
        "error" -> "No file found to upload."
      ))
    }
  }

  def delete(file: String) = Action.async { implicit request =>
    ws.url(s"${config.get[String]("server_url")}/delete")
      .withRequestTimeout(5.minutes)
      .post(Json.obj("filename" -> file))
      .map { response =>
        val deleteCount = (response.json.as[JsObject] \ "count").as[Int]
        Redirect(routes.HomeController.add())
          .flashing("success" -> s"File ${file} has been deleted (${deleteCount} chunks in total).")
      }
  }

  def feedback() = Action.async { implicit request: Request[AnyContent] =>
    val json = request.body.asJson.get.as[JsObject]
    feedbackDAO.add(Feedback(
      (json \ "chat_id").as[String],
      (json \ "message_offset").as[Int],
      (json \ "feedback").as[Boolean],
      (json \ "feedback_text").as[String]
    )).map {_ => Ok(Json.obj())}
  }

  def loadChat(chatId: String) = Action.async { implicit request: Request[AnyContent] =>
    for {
      chat <- chatDAO.get(chatId)
      messages <- chatDAO.getHistory(chatId)
      chats <- chatDAO.getLastN(20)
    } yield Ok(views.html.index(config, chats, Some((chat.get, messages))))
  }

  def deleteChat() = Action.async { implicit request: Request[AnyContent] =>
    val id = (request.body.asJson.get.as[JsObject] \ "id").as[String]
    chatDAO.delete(id).map { _ =>
      Ok(Json.obj())
    }
  }
}
