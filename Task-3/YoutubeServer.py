import pika
import json

class YouTubeServer:
    def __init__(self):
        # self.connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
        self.connection = pika.BlockingConnection(pika.ConnectionParameters('0.0.0.0'))
        self.channel = self.connection.channel()

        # Declare queues for user requests and YouTuber requests
        self.channel.queue_declare(queue='user_requests')
        self.channel.queue_declare(queue='youtuber_requests')
        
        self.youtuber_video={}
        self.user_youtuber={}
        
    def request_func(self):
        def consume_user_requests(ch, method, properties,body):
            msg=body.decode('utf-8')
            msg=json.loads(msg)
            
            if msg['type']=="SUBSCRIBE":
                if(msg['user'] in self.user_youtuber.keys()):
                    self.user_youtuber[msg['user']].append(msg['youtuber'])
                else:
                    self.user_youtuber[msg['user']]=[msg['youtuber']]
                print(f"{msg['user']} subscribed to {msg['youtuber']}")
                
            elif msg['type']=="UN-SUBSCRIBE":
                if(msg['user'] in self.user_youtuber.keys()):
                    self.user_youtuber[msg['user']].remove(msg['youtuber'])
                    print(f"{msg['user']} unsubscribed to {msg['youtuber']}")
                else:
                    print("No user found")
            elif msg['type']=="LOGIN":
                print(f"{msg['user']} logged in")       

        self.channel.basic_consume(queue='user_requests', on_message_callback=consume_user_requests, auto_ack=True)
        print('Waiting for user requests...')

        def consume_youtuber_requests(ch, method, properties,body):
            message = body.decode()
            youtuber_name, video_name = message.split()
            print(f"{youtuber_name} uploaded {video_name}")
            if(youtuber_name in self.youtuber_video.keys()):
                self.youtuber_video[youtuber_name].append(video_name)
            else:
                self.youtuber_video[youtuber_name]=[video_name]
            self.notify_users(youtuber_name, video_name)

        self.channel.basic_consume(queue='youtuber_requests', on_message_callback=consume_youtuber_requests, auto_ack=True)
        print('Waiting for YouTuber requests...')
        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            print("Server Closed")
            self.connection.close()

    def notify_users(self, youtuber_name, video_name):
        for subscriber in self.user_youtuber:
            if youtuber_name in self.user_youtuber[subscriber]:
                notification = {"youtuber": youtuber_name, "video": video_name}
                self.channel.basic_publish(exchange='', routing_key=subscriber, body=json.dumps(notification).encode('utf-8'))
                print(f"Notification sent to {subscriber}: {youtuber_name} uploaded {video_name}")

if __name__ == "__main__":
    server = YouTubeServer()
    server.request_func()
