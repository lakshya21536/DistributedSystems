import zmq

class MessagingAppServer:
    def __init__(self):
        self.groups = {} 
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://*:7777") 

    def handle_requests(self):
        while True:
            message = self.socket.recv_json() 
            if message["type"] == "JOIN_REQUEST":
                group_name=message["name"]
                group_uuid = message["group_uuid"]
                group_ip = message["ip"]
                group_port = message["port"]
                self.groups[group_uuid] = (group_ip,group_port,group_name)
                response = {"status": "SUCCESS", "message": f"SUCCESS"}
                print(f"JOIN REQUEST FROM {group_ip}:{group_port}")
                
            elif message["type"] == "GET_GROUP_LIST":
                user_port=message["port"]
                group_list = {uuid: ip_port_name for uuid, ip_port_name in self.groups.items()}
                response = {"status": "SUCCESS", "message": group_list}
                print(f"GROUP LIST REQUEST FROM {group_ip}:{user_port}")
                
            else:
                response = {"status": "ERROR", "message": "Invalid request type."}
            self.socket.send_json(response)


server = MessagingAppServer()
print("Messaging App Server started.")
server.handle_requests()