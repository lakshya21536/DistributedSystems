import zmq
import uuid
import socket

class User:
    def __init__(self, name, port):
        uuid_string = str(uuid.uuid1())
        self.uuid = uuid_string
        self.name = name
        self.ip = socket.gethostbyname(socket.gethostname())
        self.port = port

    def get_group_list(self):
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect("tcp://107.178.219.98:7777")
        # socket.connect("tcp://localhost:7777")
        message = {"type": "GET_GROUP_LIST", "port":self.port}
        socket.send_json(message)
        response = socket.recv_json()
        if response["status"] == "SUCCESS":
            group_list=response["message"]
            for uuid,ip_port_name in group_list.items():
                print(f'{ip_port_name[2]} - {ip_port_name[0]}:{ip_port_name[1]}')
        else:
            print("Failed to register Group Server with Message Server.")

    def join_group(self):
        group_port=input("Enter port of the group : ")
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect(f"tcp://34.28.190.155:{group_port}")
        # socket.connect(f"tcp://localhost:{group_port}")
        message = {"type": "JOIN_GROUP", "user_uuid": self.uuid}
        socket.send_json(message)
        response = socket.recv_json()
        if response["status"] == "SUCCESS":
            print(response["message"])
        else:
            print("Failed")

    def leave_group(self):
        group_port=input("Enter port of the group : ")
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect(f"tcp://34.28.190.155:{group_port}")
        # socket.connect(f"tcp://localhost:{group_port}")
        message = {"type": "LEAVE_GROUP", "user_uuid": self.uuid}
        socket.send_json(message)
        response = socket.recv_json()
        if response["status"] == "SUCCESS":
            print(response["message"])
        else:
            print("Failed")

    def send_message(self):
        group_port=input("Enter port of the group : ")
        message_content=input("Enter the complete message : ")
        message = {"type": "SEND_MESSAGE", "user_uuid": self.uuid,"content": message_content}
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect(f"tcp://34.28.190.155:{group_port}")
        # socket.connect(f"tcp://localhost:{group_port}")
        socket.send_json(message)
        response = socket.recv_json()
        if response["status"] == "SUCCESS":
            print("SUCCESS")
        else:
            print("FAILED")

    def get_messages(self):
        group_port=input("Enter port of the group : ")
        timestamp=input("Enter the timestamp : ")
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect(f"tcp://34.28.190.155:{group_port}")
        # socket.connect(f"tcp://localhost:{group_port}")
        if(timestamp==''):
            message = {"type": "GET_MESSAGES", "user_uuid":self.uuid}
        else:
            message = {"type": "GET_MESSAGES", "user_uuid":self.uuid, "timestamp": timestamp}
        socket.send_json(message)
        response = socket.recv_json()
        if(response["status"]=="SUCCESS"):
            if len(response["messages"])==0:
                print("No messages found")
            else:
                for i in response["messages"]:
                    print(i['user_uuid'],i['content'],i['timestamp'])
        else:
            print(response["message"])

if __name__ == "__main__":
    name=input("Enter your name : ")
    port=input("Enter your port number : ")
    user=User(name,port)
    while(True):
        print('''Choose 
1. Request message server to get group list
2. Join group
3. Leave group
4. Get message
5. Send message
6. Exit''')
        inp=int(input("Enter your choice : "))
        if(inp==1):
            user.get_group_list()
        elif(inp==2):
            user.join_group()
        elif(inp==3):
            user.leave_group()
        elif(inp==4):
            user.get_messages()
        elif(inp==5):
            user.send_message()
        elif(inp==6):
            break
        else:
            print("Invalid input. Try again")