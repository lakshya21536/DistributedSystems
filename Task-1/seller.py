import grpc
import market_pb2
import market_pb2_grpc
import time
from concurrent import futures
import threading
import uuid

class SellerClient:
    def __init__(self, server_address):
        self.server_address = server_address
        self.channel = grpc.insecure_channel(self.server_address)
        self.stub = market_pb2_grpc.MarketStub(self.channel)
        self.c_item = None
        self.notification_thread = threading.Thread(target=self.t_func, daemon=True)
        self.notification_thread.start()  # Start the notification thread when the client is initialized
        self.uuid = str(uuid.uuid1())

    def t_func(self):
        while True:
            if self.c_item:
                print(self.c_item)
            self.c_item = None

    
    def register_seller(self, address):
        request = market_pb2.RegisterSellerRequest(address=address, uuid=self.uuid)
        response = self.stub.RegisterSeller(request)
        print("Response:", response.message)
        print()

    def sell_item(self, name, category, quantity, description, seller_address, price):
        request = market_pb2.SellItemRequest(
            name=name,
            category=category,
            quantity=quantity,
            description=description,
            seller_address=seller_address,
            price=price,
            seller_uuid=self.uuid
        )
        response = self.stub.SellItem(request)
        print("Response:", response.message)
        print()

    def update_item(self, item_id, price, quantity, seller_address):
        request = market_pb2.UpdateItemRequest(
            id = item_id,
            price = price,
            quantity = quantity,
            seller_address = seller_address,
            seller_uuid = self.uuid
        )
        response = self.stub.UpdateItem(request)
        print("Response:", response.message)
        print()

    def delete_item(self, item_id, seller_address):
        request = market_pb2.DeleteItemRequest(
            id = item_id,
            seller_address = seller_address,
            seller_uuid = self.uuid
        )
        response = self.stub.DeleteItem(request)
        print("Response:", response.message)
        print()

    def display_seller_items(self, seller_address):
        request = market_pb2.DisplaySellerItemsRequest(
            seller_address=seller_address,
            seller_uuid=self.uuid
        )
        response = self.stub.DisplaySellerItems(request)
        if response.message:
            print("Error:", response.message)
            print()
        else:
            if(len(response.items) == 0):
                print("No items to display\n")
                print()
            else:
                for item in response.items:
                    print("Item ID:", item.id)
                    print("Name:", item.name)
                    print("Category:", item.category)
                    print("Quantity:", item.quantity)
                    print("Description:", item.description)
                    print("Seller Address:", item.seller_address)
                    print("Price:", item.price)
                    print("Rating:", item.average_rating)
                    print() 

    def notify_client(self, role, item_info, address):
        request = market_pb2.NotifyClientRequest(
            role=role,
            item_info=item_info,
            address=address
        )
        response = self.stub.NotifyClient(request)
        print("Notification response:", response.message)

    def RecvNotification(self, request, context):
        item = request.item
        print(item)
        return market_pb2.RecvNotificationResponse(message = "SUCCESS")
    # def NotifyOther(self, request, context):
    #     try:
    #         item_info = request.item
    #         print("Notificaction Receives : " , item_info)
    #         return market_pb2.NotifyOtherResponse(message = "Notifcation successfully sent!!")
    #     except:
    #         print("Nhi chlra")

    def close_channel(self):
        self.channel.close()

# class SellerServicer(market_pb2_grpc.SellerServicer):
#     def __init__(self):
#         pass

#     def RecvNotification(self, request):
#         item = request.item
#         print(item)
#         return market_pb2.RecvNotificationResponse(message = "chl gya bc please chl jaa")

def print_menu():
    print("1. Register Seller")
    print("2. Sell Item")
    print("3. Update Item")
    print("4. Delete Item")
    print("5. Display Seller Items")
    print("6. Exit")

def start_server(port_no):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    market_pb2_grpc.add_SellerServicer_to_server(SellerClient('104.197.253.159:50051'), server)
    server.add_insecure_port(f'[::]:{port_no}')
    server.start()
    try:
        while True:
            time.sleep(86400) 
    except KeyboardInterrupt:
        server.stop(0)  


market_server_address = "104.197.253.159:50051"
port_no = input("Enter port number: ")
seller_client = SellerClient(market_server_address)
thread = threading.Thread(target = start_server , args=(port_no,), daemon=True)
thread.start()
while True:
    print_menu()
    ip="34.41.83.118:"
    choice = input("Enter your choice: ")

    if choice == "1":
        address = input("Enter seller port: ")
        # uuid = input("Enter seller UUID: ")
        seller_client.register_seller(ip+address)
    elif choice == "2":
        name = input("Enter item name: ")
        category = input("Enter item category: ")
        quantity = int(input("Enter quantity: "))
        description = input("Enter item description: ")
        seller_address = input("Enter seller port: ")
        price = float(input("Enter item price: "))
        # seller_uuid = input("Enter seller UUID: ")
        seller_client.sell_item(name, category, quantity, description,ip+seller_address, price)
    elif choice == "3":
        item_id = int(input("Enter item ID: "))
        price = float(input("Enter new price: "))
        quantity = int(input("Enter new quantity: "))
        seller_address = input("Enter seller port: ")
        # seller_uuid = input("Enter seller UUID: ")
        seller_client.update_item(item_id, price, quantity,ip+seller_address)
    elif choice == "4":
        item_id = int(input("Enter item ID: "))
        seller_address = input("Enter seller port: ")
        # seller_uuid = input("Enter seller UUID: ")
        seller_client.delete_item(item_id,ip+seller_address)
    elif choice == "5":
        seller_address = input("Enter seller port: ")
        # seller_uuid = input("Enter seller UUID: ")
        seller_client.display_seller_items(ip+seller_address)
    elif choice == "6":
        seller_client.close_channel()
        break
    else:
        print("Invalid choice. Please try again.")

