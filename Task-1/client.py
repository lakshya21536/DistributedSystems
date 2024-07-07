import grpc
import market_pb2
import market_pb2_grpc
import time
from concurrent import futures
import threading

class Client:
    def __init__(self, server_address):
        self.server_address = server_address
        self.channel = grpc.insecure_channel(self.server_address)
        self.stub = market_pb2_grpc.MarketStub(self.channel)

    def search_item(self, item_name, item_category):
        request = market_pb2.SearchItemRequest(
            name = item_name,
            category = item_category
        )
        response = self.stub.SearchItem(request)
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

    def buy_item(self, item_id, quantity, buyer_address):
        request = market_pb2.BuyItemRequest(
            id = item_id,
            quantity = quantity,
            buyer_address = buyer_address
        )
        response = self.stub.BuyItem(request)
        
        print("Response:", response.message)

    def add_to_wishlist(self, item_id, buyer_address):
        request = market_pb2.AddToWishListRequest(
            id = item_id,
            buyer_address = buyer_address
        )
        response = self.stub.AddToWishList(request)
        print("Response:", response.message)

    def rate_item(self, item_id, rating, buyer_address):
        request = market_pb2.RateItemRequest(
            item_id = item_id,
            rating = rating,
            buyer_address = buyer_address
        )
        response = self.stub.RateItem(request)
        print("Response:", response.message)

    def notify_client(self, item_info, seller_address):
        request = market_pb2.NotifyClientRequest(
            role = "seller",
            item_info = item_info,
            address = seller_address
        )
        response = self.stub.NotifyClient(request)
        print("Response:", response.message)

    def RecieveNotification(self, request, context):
        item = request.item
        print(item)
        return market_pb2.RecieveNotificationResponse(message = "SUCCESS")

    def close_channel(self):
        self.channel.close()

def print_menu():
    print("1. Search Item")
    print("2. Buy Item")
    print("3. Add Item to Wishlist")
    print("4. Rate Item")
    print("5. Exit")

def start_server(port_no):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    market_pb2_grpc.add_ClientServicer_to_server(Client('104.197.253.159:50051'), server)
    server.add_insecure_port(f'[::]:{port_no}')
    server.start()
    try:
        while True:
            time.sleep(86400) 
    except KeyboardInterrupt:
        server.stop(0)  

if __name__ == "__main__":
    market_server_address = "104.197.253.159:50051"
    client = Client(market_server_address)
    port_no = input("Enter port no: ")
    thread = threading.Thread(target = start_server, args=(port_no,) , daemon=True)
    thread.start()
    while True:
        print_menu()
        choice = input("Enter your choice: ")
        ip="35.192.113.44:" 
        if choice == "1":
            item_name = input("Enter item name: ")
            item_category = input("Enter item category: ")
            client.search_item(item_name, item_category)
        elif choice == "2":
            item_id = int(input("Enter item ID: "))
            quantity = int(input("Enter quantity: "))
            buyer_address = input("Enter buyer port: ")
            client.buy_item(item_id, quantity, ip+buyer_address)
        elif choice == "3":
            item_id = int(input("Enter item ID: "))
            buyer_address = input("Enter buyer port: ")
            client.add_to_wishlist(item_id,ip+buyer_address)
        elif choice == "4":
            item_id = int(input("Enter item ID: "))
            rating = int(input("Enter rating: "))
            buyer_address = input("Enter buyer port: ")
            client.rate_item(item_id, rating,ip+buyer_address)
        elif choice == "5":
            client.close_channel()
            break
        else:
            print("Invalid choice. Please try again.\n")
