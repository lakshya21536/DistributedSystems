import grpc
from concurrent import futures
import time
import market_pb2
import market_pb2_grpc
import uuid

class MarketServicer(market_pb2_grpc.MarketServicer):
    def __init__(self):
        self.seller_registry = {}
        self.item_registry = {}
        self.item_id_counter = 1
        self.wishlist_registry = {}
        self.rating_registry = {}
        self.itemChanged = False
        self.notiItem = None
        self.tot_num = {}
        
    def RegisterSeller(self, request, context):
        address = request.address
        uuid = request.uuid
        print(f"Seller join request from {address}, uuid = {uuid}\n")

        if address in self.seller_registry.values():
            return market_pb2.RegisterSellerResponse(message="FAILED: Seller already registered with this address")
        
        self.seller_registry[uuid] = address
        return market_pb2.RegisterSellerResponse(message="SUCCESS: Seller Registered!")

    def SellItem(self, request, context):
        name = request.name
        category = request.category
        quantity = request.quantity
        description = request.description
        seller_address = request.seller_address
        price = request.price
        seller_uuid = request.seller_uuid
        print(f"Sell item request from {seller_address}\n")

        if seller_uuid not in self.seller_registry:
            return market_pb2.SellItemResponse(message="FAILED: Invalid seller UUID")

        if category not in ["ELECTRONICS", "FASHION", "OTHERS"]:
            return market_pb2.SellItemResponse(message="FAILED: Invalid category")

        if quantity <= 0:
            return market_pb2.SellItemResponse(message="FAILED: Invalid quantity")

        if price < 0:
            return market_pb2.SellItemResponse(message="FAILED: Invalid price")
        
        item_id = self.item_id_counter
        self.item_id_counter += 1

        item = market_pb2.Item(
            id=item_id,
            name=name,
            category=category,
            quantity=quantity,
            description=description,
            seller_address=seller_address,
            price=price,
            average_rating=0.0
        )
        self.item_registry[item_id] = item

        return market_pb2.SellItemResponse(message=f"SUCCESS your item_id is: {item_id}", item_id=item_id)

    def UpdateItem(self, request, context):
        item_id = request.id
        price = request.price
        quantity = request.quantity
        seller_address = request.seller_address
        seller_uuid = request.seller_uuid
        print(f"Update Item {item_id} request from {seller_address}\n")

        if seller_uuid not in self.seller_registry:
            return market_pb2.UpdateItemResponse(message="FAILED: Invalid seller UUID")

        if item_id not in self.item_registry:
            return market_pb2.UpdateItemResponse(message=f"FAILED: Item ID {item_id} not found")

        item = self.item_registry[item_id]

        if item.seller_address != seller_address:
            return market_pb2.UpdateItemResponse(message="FAILED: Unauthorized to update this item")

        if price >= 0:
            item.price = price
        else:
            return market_pb2.UpdateItemResponse(message="FAILED: Invalid price")

        if quantity >= 0:
            item.quantity = quantity
        else:
            return market_pb2.UpdateItemResponse(message="FAILED: Invalid quantity")
        
        self.item_registry[item_id] = item

        #NotifyClient
        # print(self.wishlist_registry)
        for buyer_add in self.wishlist_registry:
            sets = self.wishlist_registry[buyer_add]

            if item_id in sets:
                channel1 = grpc.insecure_channel(buyer_add)
                stub = market_pb2_grpc.ClientStub(channel1)

                request = market_pb2.RecieveNotificationRequest(
                    item = item
                )
                try:
                    response = stub.RecieveNotification(request)
                    print("Notification sent to buyer:", response.message)
                except grpc.RpcError as e:
                    print("Failed to send notification to buyer:", e.details())

        return market_pb2.UpdateItemResponse(message="SUCCESS")

    def DeleteItem(self, request, context):
        item_id = request.id
        seller_address = request.seller_address
        seller_uuid = request.seller_uuid
        print(f"Delete Item {item_id} request from {seller_address}\n")

        if seller_uuid not in self.seller_registry:
            return market_pb2.DeleteItemResponse(message="FAILED: Invalid seller UUID")

        if item_id not in self.item_registry:
            return market_pb2.DeleteItemResponse(message=f"FAILED: Item ID {item_id} not found")

        item = self.item_registry[item_id]

        if item.seller_address != seller_address:
            return market_pb2.DeleteItemResponse(message="FAILED: Unauthorized to delete this item")

        del self.item_registry[item_id]

        return market_pb2.DeleteItemResponse(message="SUCCESS")

    def DisplaySellerItems(self, request, context):
        seller_address = request.seller_address
        seller_uuid = request.seller_uuid
        print(f"Display Items request from {seller_address}\n")

        if seller_uuid not in self.seller_registry:
            return market_pb2.DisplaySellerItemsResponse(message="FAILED: Invalid seller UUID")

        seller_items = [item for item in self.item_registry.values() if item.seller_address == seller_address]

        items_list = []
        for item in seller_items:
            item_message = market_pb2.Item(
                id=item.id,
                name=item.name,
                category=item.category,
                quantity=item.quantity,
                description=item.description,
                seller_address=item.seller_address,
                price=item.price,
                average_rating=item.average_rating  
            )
            items_list.append(item_message)

        return market_pb2.DisplaySellerItemsResponse(items=items_list)

    def SearchItem(self, request, context):
        name = request.name
        item_category = request.category
        print(f"Search request for item name: {name}, Category: {item_category}\n")
        matched_items = []
        if(name == '' and item_category == 'ANY'):
            for item in self.item_registry.values():
                matched_items.append(item)
        elif(name == ''):
            for item in self.item_registry.values():
                if(item_category == item.category):
                    matched_items.append(item)
        else:
            for item in self.item_registry.values():
                if((item_category == "ANY" or item.category == item_category)):
                    matched_items.append(item)

        items_list = []
        for item in matched_items:
            item_message = market_pb2.Item(
                id=item.id,
                name=item.name,
                category=item.category,
                quantity=item.quantity,
                description=item.description,
                seller_address=item.seller_address,
                price=item.price,
                average_rating=item.average_rating
            )
            items_list.append(item_message)

        return market_pb2.SearchItemResponse(items=items_list)

    def BuyItem(self, request, context):
        item_id = request.id
        quantity = request.quantity
        buyer_address = request.buyer_address
        print(f"Buy request {quantity} of Item {item_id}, from {buyer_address}\n")

        if item_id not in self.item_registry:
            return market_pb2.BuyItemResponse(message="FAILED: Item not found")

        item = self.item_registry[item_id]

        if item.quantity < quantity:
            return market_pb2.BuyItemResponse(message="FAILED: Insufficient quantity available")
        
        item.quantity -= quantity

        #NotifyClient
        server_address = item.seller_address
        channel1 = grpc.insecure_channel(server_address)
        stub = market_pb2_grpc.SellerStub(channel1)

        request = market_pb2.RecvNotificationRequest(
            item = item
        )
        try:
            response = stub.RecvNotification(request)
            print("Notification sent to seller:", response.message)
        except grpc.RpcError as e:
            print("Failed to send notification to seller:", e.details())
        return market_pb2.BuyItemResponse(message = "SUCCESS")
        
        # seller_address = item.seller_address
        # notification_request = market_pb2.NotifyClientRequest(
        #     role="seller",
        #     item_info=item,
        #     address=seller_address
        # )
        # self.NotifyClient(notification_request, context)
        
        # return market_pb2.BuyItemResponse(message="SUCCESS", item_info = item)

    def AddToWishList(self, request, context):
        item_id = request.id
        buyer_address = request.buyer_address
        print(f"Wishlist request of Item {item_id}, from {buyer_address}\n")

        if item_id not in self.item_registry:
            return market_pb2.AddToWishListResponse(message="FAILED: Item not found")

        if buyer_address not in self.wishlist_registry:
            self.wishlist_registry[buyer_address] = set()

        self.wishlist_registry[buyer_address].add(item_id)

        return market_pb2.AddToWishListResponse(message="SUCCESS")

    def RateItem(self, request, context):
        item_id = request.item_id
        rating = request.rating
        buyer_address = request.buyer_address
        print(f"{buyer_address} rated Item {item_id} with {rating} stars\n")

        if(rating > 5 or rating < 1):
            return market_pb2.RateItemResponse(message = "FAILED: Invalid Rating")
        if item_id not in self.item_registry:
            return market_pb2.RateItemResponse(message="FAILED: Item not found")

        if item_id in self.rating_registry and buyer_address in self.rating_registry[item_id]:
            return market_pb2.RateItemResponse(message="FAILED: Buyer already rated this item")

        if item_id not in self.rating_registry:
            self.rating_registry[item_id] = {}

        self.rating_registry[item_id][buyer_address] = rating
        if(item_id in self.tot_num):
            self.item_registry[item_id].average_rating = ((self.item_registry[item_id].average_rating * self.tot_num[item_id]) + rating)/(self.tot_num[item_id]+1)
            self.tot_num[item_id] += 1
        else:
            self.tot_num[item_id] = 0
            self.item_registry[item_id].average_rating = ((self.item_registry[item_id].average_rating * self.tot_num[item_id]) + rating)/(self.tot_num[item_id]+1)
            self.tot_num[item_id] += 1

        return market_pb2.RateItemResponse(message="SUCCESS")

    # def NotifyClient(self, request, context):
    #     role = request.role
    #     address = request.address
    #     item_info = request.item_info
    #     if(self.notiItem):
    #         print("Sending notification to seller")
    #         return market_pb2.NotifyClientResponse(message = "Notification sent successfully", item = self.notiItem)

    # def NotifyClient(self, request, context):
    #     role = request.role
    #     address = request.address
    #     item_info = request.item_info

    #     server_address = address
    #     channel = grpc.insecure_channel(server_address)
    #     stub = market_pb2_grpc.SellerStub(channel)

    #     try:
    #         response = stub.NotifyOther(request)
    #         print("Notification sent to seller:", response.message)
    #     except grpc.RpcError as e:
    #         print("Failed to send notification to seller:", e.details())

    #     channel.close()
    # def NotifySeller(self, request, context):
    #     address = 

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    market_pb2_grpc.add_MarketServicer_to_server(MarketServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(86400)  # One day in seconds
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()
