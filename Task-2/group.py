import zmq
from datetime import datetime
from threading import Thread
import uuid
import queue
import socket 

class GroupServer:
    def __init__(self,name,port):
        uuid_string = str(uuid.uuid1())
        self.uuid = uuid_string
        self.name=name
        self.ip = socket.gethostbyname(socket.gethostname())
        self.port = port
        self.users = set() 
        self.messages = []  
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{self.port}")

    def handle_get_msg(self,message,q):
        print(f"MESSAGE REQUEST FROM {message['user_uuid']}")
        user_uuid = message["user_uuid"]
        response=None
        if user_uuid not in self.users:
            print("User is not the member of the group")
            response = {"status": "FAILED", "message":"You are not the member of this group."}
        else:
            timestamp = ""
            if "timestamp" in message.keys():
                timestamp = message["timestamp"]
                format = "%Y-%m-%d %H:%M:%S" 
                timestamp = datetime.strptime(timestamp, format)
                user_messages = [msg for msg in self.messages if datetime.strptime(msg["timestamp"], format) >= timestamp]    
            else:
                user_messages = [msg for msg in self.messages]
            response = {"status": "SUCCESS", "messages": user_messages}
        q.put(response)
    
    def handle_send_msg(self,message,q):
        print(f"MESSAGE SEND FROM {message['user_uuid']}")
        user_uuid = message["user_uuid"]
        content = message["content"]
        response=None
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if user_uuid in self.users:
            self.messages.append({"user_uuid": user_uuid, "content": content, "timestamp": timestamp})
            response = {"status": "SUCCESS", "message": "Message sent successfully."}
        else:
            print("User not member")
            response={"status": "FAILED", "message": "User is not the member of the group"}
        q.put(response)

    def handle_requests(self):
        while True:
            message = self.socket.recv_json()
            if message["type"] == "JOIN_GROUP":
                user_uuid = message["user_uuid"]
                self.users.add(user_uuid)
                response = {"status": "SUCCESS", "message": "SUCCESS"}
                print(f"JOIN REQUEST FROM {user_uuid}")
                
            elif message["type"] == "LEAVE_GROUP":
                user_uuid = message["user_uuid"]
                if user_uuid in self.users:
                    self.users.remove(user_uuid)
                    response = {"status": "SUCCESS", "message": f"SUCCESS"}
                    print(f"LEAVE REQUEST FROM {user_uuid}")
                else:
                    response = {"status": "ERROR", "message":"FAILED"}
                    print(f"User {user_uuid} is not in the group.")
                    
            elif message["type"] == "SEND_MESSAGE":
                q=queue.Queue()
                th=Thread(target=self.handle_send_msg,args=(message,q))
                th.start()
                th.join()
                response=q.get()
                
            elif message["type"] == "GET_MESSAGES":
                q=queue.Queue()
                th=Thread(target=self.handle_get_msg,args=(message,q))
                th.start()
                th.join()
                response=q.get()
                
            else:
                response = {"status": "ERROR", "message": "Invalid request type."}

            self.socket.send_json(response)
    
    def register_with_message_server(self):
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect("tcp://107.178.219.98:7777")
        # socket.connect("tcp://localhost:7777")
        message = {"type": "JOIN_REQUEST", "name": self.name, "ip": self.ip, "port": self.port,"group_uuid":self.uuid}
        socket.send_json(message)
        response = socket.recv_json()  
        if response["status"] == "SUCCESS":
            print(response['message']) 
        else:
            print("Failed to register Group Server with Message Server.")

if __name__ == "__main__":
    name=input("Enter name of the group : ")
    port=input("Enter the port number : ")
    server = GroupServer(name,port)
    print("Group Server started.")
    server.register_with_message_server()
    server.handle_requests()
