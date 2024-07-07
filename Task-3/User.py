import sys
import pika
import json

class User:
    def __init__(self, username):
        self.username = username
        self.connection = pika.BlockingConnection(pika.ConnectionParameters('107.178.219.98'))
        self.channel = self.connection.channel()

        # Declare queue for receiving notifications
        self.channel.queue_declare(queue=username)

    def updateSubscription(self, youtuber, action):
        if action=='s':
            message = {
                "type": "SUBSCRIBE",
                "user": self.username,
                "youtuber": youtuber,
            }
        elif action=='u':
            message = {
                "type": "UN-SUBSCRIBE",
                "user": self.username,
                "youtuber": youtuber,
            }
        msg=json.dumps(message).encode('utf-8')
        self.channel.basic_publish(exchange='', routing_key='user_requests', body=msg)
        print("SUCCESS: Subscription updated successfully.")

    def receiveNotifications(self):
        def callback(ch, method, properties, body):
            message=body.decode('utf-8')
            message=json.loads(message)
            print(f"New notification : {message['youtuber']} uploaded {message['video']}")
        self.channel.basic_consume(queue=self.username, on_message_callback=callback, auto_ack=True)
        print("Waiting for notifications...")
        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            print("Server Closed")
            self.connection.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python User.py <Username> [s/u <YouTuberName>]")
        sys.exit(1)

    username = sys.argv[1]
    user = User(username)

    if len(sys.argv) == 4:
        action = sys.argv[2]
        youtuber = sys.argv[3]
        if action == 's' or action=='u':
            user.updateSubscription(youtuber,action)
        else:
            print("Invalid action. Please use 's' to subscribe or 'u' to unsubscribe.")
            sys.exit(1)
    else:
        message = {
            "type": "LOGIN",
            "user": username,
        }
        msg=json.dumps(message).encode('utf-8')
        user.channel.basic_publish(exchange='', routing_key='user_requests', body=msg)
    user.receiveNotifications()