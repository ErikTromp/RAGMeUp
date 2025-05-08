package models.daos

import models.{Chat, ChatMessage, SQLTables}
import play.api.Configuration
import play.api.db.slick.{DatabaseConfigProvider, HasDatabaseConfigProvider}
import play.db.NamedDatabase
import slick.jdbc.JdbcProfile

import javax.inject.{Inject, Singleton}
import scala.concurrent.{ExecutionContext, Future}

@Singleton
class ChatDAO @Inject()(@NamedDatabase("ragmeup") protected val dbConfigProvider: DatabaseConfigProvider, sqlTables: SQLTables)
                           (implicit ec: ExecutionContext, config: Configuration) extends HasDatabaseConfigProvider[JdbcProfile] {

  import profile.api._
  import sqlTables._

  def getLastN(n: Int): Future[Seq[Chat]] = db.run(Chats.sortBy(_.createdAt.desc).take(n).result)
  def get(id: String): Future[Option[Chat]] = db.run(Chats.filter(_.id === id).result.headOption)
  def add(chat: Chat)(implicit ec: ExecutionContext): Future[Chat] = {
    val insertAndReturn = db.run((Chats returning Chats) += chat)
    insertAndReturn.recoverWith {
      case ex =>
        // SQLite does not support returning, so fall back to insert and fetch
        val insertAction = Chats += chat
        val fetchAction = Chats.filter(_.id === chat.id).result.head

        db.run(insertAction).flatMap(_ => db.run(fetchAction))
    }
  }
  def addChatMessage(message: ChatMessage): Future[Int] = db.run(ChatMessages += message)
  def delete(id: String): Future[Int] = db.run(Chats.filter(_.id === id).delete)
  def getHistory(chatId: String): Future[Seq[ChatMessage]] = db.run(
    Chats
      .join(ChatMessages)
      .on(_.id === _.chatId)
      .filter(chatWithMessage => chatWithMessage._1.id ===chatId)
      .map(_._2)
      .sortBy(_.messageOffset)
      .result
  )

}