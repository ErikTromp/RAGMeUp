package models

import play.api.Configuration
import play.api.db.slick.{DatabaseConfigProvider, HasDatabaseConfigProvider}
import play.db.NamedDatabase
import slick.jdbc.{JdbcProfile, PostgresProfile, SQLiteProfile}

import javax.inject.{Inject, Singleton}

case class Chat(id: String, title: String, createdAt: Long)
case class ChatMessage(chatId: String, messageOffset: Int, createdAt: Long, text: String, role: String,
                       documents: String, rewritten: Option[String], fetchedNewDocuments: Boolean)
case class Feedback(
 chatId: String, messageOffset: Int, feedback: Boolean, feedbackText: String
)

@Singleton
class SQLTables @Inject() (@NamedDatabase("ragmeup") protected val dbConfigProvider: DatabaseConfigProvider, config: Configuration)
  extends HasDatabaseConfigProvider[JdbcProfile] {

  val selectedProfile: JdbcProfile = config.get[String]("slick.dbs.ragmeup.profile") match {
    case "slick.jdbc.SQLiteProfile$" => SQLiteProfile
    case "slick.jdbc.PostgresProfile$" => PostgresProfile
    case other =>
      throw new IllegalArgumentException(s"Unsupported database profile: $other")
  }

  import selectedProfile.api._

  class ChatTable(tag: Tag) extends Table[Chat](tag, "chats") {
    def id = column[String]("id", O.PrimaryKey)
    def title = column[String]("title")
    def createdAt = column[Long]("created_at")
    def * = (id, title, createdAt) <> (Chat.tupled, Chat.unapply)
  }

  val Chats = TableQuery[ChatTable]

  class ChatMessageTable(tag: Tag) extends Table[ChatMessage](tag, "chat_messages") {
    def chatId = column[String]("chat_id")
    def messageOffset = column[Int]("message_offset")
    def createdAt = column[Long]("created_at")
    def text = column[String]("text")
    def role = column[String]("role")
    def documents = column[String]("documents")
    def rewritten = column[Option[String]]("rewritten")
    def fetchedNewDocuments = column[Boolean]("fetched_new_documents")
    def * = (chatId, messageOffset, createdAt, text, role, documents, rewritten, fetchedNewDocuments) <> (ChatMessage.tupled, ChatMessage.unapply)
    def chat = foreignKey("chat_fk", chatId, TableQuery[ChatTable])(_.id)
  }

  val ChatMessages = TableQuery[ChatMessageTable]

  class FeedbackTable(tag: Tag) extends Table[Feedback](tag, "feedback") {
    def chatId = column[String]("chat_id")
    def messageOffset = column[Int]("message_offset")
    def feedback = column[Boolean]("feedback")
    def feedbackText = column[String]("feedback_text")
    def * = (chatId, messageOffset, feedback, feedbackText).mapTo[Feedback]
  }

  val Feedbacks = TableQuery[FeedbackTable]
}
