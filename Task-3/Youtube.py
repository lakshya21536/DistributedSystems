import sys
import pika

class Youtuber:
    def __init__(self):
        self.connection = pika.BlockingConnection(pika.ConnectionParameters('107.178.219.98'))
        self.channel = self.connection.channel()

        # Declare queue for sending video to YouTube server
        self.channel.queue_declare(queue='youtuber_requests')

    def publishVideo(self, youtuber, videoName):
        message = f"{youtuber} {videoName}"
        self.channel.basic_publish(exchange='', routing_key='youtuber_requests', body=message)
        print("SUCCESS: Video published successfully.")

if __name__ == "__main__":
    if len(sys.argv)<3:
        print("Usage: python Youtuber.py <YoutuberName> <VideoName>")
        sys.exit(1)

    youtuber_name = sys.argv[1]
    video_name = ' '.join(sys.argv[2:])

    # Run the Youtuber service to publish a video
    youtuber = Youtuber()
    youtuber.publishVideo(youtuber_name, video_name)