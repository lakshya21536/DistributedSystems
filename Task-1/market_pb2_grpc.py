# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import market_pb2 as market__pb2


class MarketStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.RegisterSeller = channel.unary_unary(
                '/Market/RegisterSeller',
                request_serializer=market__pb2.RegisterSellerRequest.SerializeToString,
                response_deserializer=market__pb2.RegisterSellerResponse.FromString,
                )
        self.SellItem = channel.unary_unary(
                '/Market/SellItem',
                request_serializer=market__pb2.SellItemRequest.SerializeToString,
                response_deserializer=market__pb2.SellItemResponse.FromString,
                )
        self.UpdateItem = channel.unary_unary(
                '/Market/UpdateItem',
                request_serializer=market__pb2.UpdateItemRequest.SerializeToString,
                response_deserializer=market__pb2.UpdateItemResponse.FromString,
                )
        self.DeleteItem = channel.unary_unary(
                '/Market/DeleteItem',
                request_serializer=market__pb2.DeleteItemRequest.SerializeToString,
                response_deserializer=market__pb2.DeleteItemResponse.FromString,
                )
        self.DisplaySellerItems = channel.unary_unary(
                '/Market/DisplaySellerItems',
                request_serializer=market__pb2.DisplaySellerItemsRequest.SerializeToString,
                response_deserializer=market__pb2.DisplaySellerItemsResponse.FromString,
                )
        self.SearchItem = channel.unary_unary(
                '/Market/SearchItem',
                request_serializer=market__pb2.SearchItemRequest.SerializeToString,
                response_deserializer=market__pb2.SearchItemResponse.FromString,
                )
        self.BuyItem = channel.unary_unary(
                '/Market/BuyItem',
                request_serializer=market__pb2.BuyItemRequest.SerializeToString,
                response_deserializer=market__pb2.BuyItemResponse.FromString,
                )
        self.AddToWishList = channel.unary_unary(
                '/Market/AddToWishList',
                request_serializer=market__pb2.AddToWishListRequest.SerializeToString,
                response_deserializer=market__pb2.AddToWishListResponse.FromString,
                )
        self.RateItem = channel.unary_unary(
                '/Market/RateItem',
                request_serializer=market__pb2.RateItemRequest.SerializeToString,
                response_deserializer=market__pb2.RateItemResponse.FromString,
                )
        self.NotifyClient = channel.unary_unary(
                '/Market/NotifyClient',
                request_serializer=market__pb2.NotifyClientRequest.SerializeToString,
                response_deserializer=market__pb2.NotifyClientResponse.FromString,
                )
        self.GetItemRatings = channel.unary_unary(
                '/Market/GetItemRatings',
                request_serializer=market__pb2.GetItemRatingsRequest.SerializeToString,
                response_deserializer=market__pb2.GetItemRatingsResponse.FromString,
                )
        self.NotifySeller = channel.unary_unary(
                '/Market/NotifySeller',
                request_serializer=market__pb2.NotifySellerRequest.SerializeToString,
                response_deserializer=market__pb2.NotifySellerResponse.FromString,
                )


class MarketServicer(object):
    """Missing associated documentation comment in .proto file."""

    def RegisterSeller(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SellItem(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def UpdateItem(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DeleteItem(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DisplaySellerItems(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SearchItem(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def BuyItem(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def AddToWishList(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def RateItem(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def NotifyClient(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetItemRatings(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def NotifySeller(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_MarketServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'RegisterSeller': grpc.unary_unary_rpc_method_handler(
                    servicer.RegisterSeller,
                    request_deserializer=market__pb2.RegisterSellerRequest.FromString,
                    response_serializer=market__pb2.RegisterSellerResponse.SerializeToString,
            ),
            'SellItem': grpc.unary_unary_rpc_method_handler(
                    servicer.SellItem,
                    request_deserializer=market__pb2.SellItemRequest.FromString,
                    response_serializer=market__pb2.SellItemResponse.SerializeToString,
            ),
            'UpdateItem': grpc.unary_unary_rpc_method_handler(
                    servicer.UpdateItem,
                    request_deserializer=market__pb2.UpdateItemRequest.FromString,
                    response_serializer=market__pb2.UpdateItemResponse.SerializeToString,
            ),
            'DeleteItem': grpc.unary_unary_rpc_method_handler(
                    servicer.DeleteItem,
                    request_deserializer=market__pb2.DeleteItemRequest.FromString,
                    response_serializer=market__pb2.DeleteItemResponse.SerializeToString,
            ),
            'DisplaySellerItems': grpc.unary_unary_rpc_method_handler(
                    servicer.DisplaySellerItems,
                    request_deserializer=market__pb2.DisplaySellerItemsRequest.FromString,
                    response_serializer=market__pb2.DisplaySellerItemsResponse.SerializeToString,
            ),
            'SearchItem': grpc.unary_unary_rpc_method_handler(
                    servicer.SearchItem,
                    request_deserializer=market__pb2.SearchItemRequest.FromString,
                    response_serializer=market__pb2.SearchItemResponse.SerializeToString,
            ),
            'BuyItem': grpc.unary_unary_rpc_method_handler(
                    servicer.BuyItem,
                    request_deserializer=market__pb2.BuyItemRequest.FromString,
                    response_serializer=market__pb2.BuyItemResponse.SerializeToString,
            ),
            'AddToWishList': grpc.unary_unary_rpc_method_handler(
                    servicer.AddToWishList,
                    request_deserializer=market__pb2.AddToWishListRequest.FromString,
                    response_serializer=market__pb2.AddToWishListResponse.SerializeToString,
            ),
            'RateItem': grpc.unary_unary_rpc_method_handler(
                    servicer.RateItem,
                    request_deserializer=market__pb2.RateItemRequest.FromString,
                    response_serializer=market__pb2.RateItemResponse.SerializeToString,
            ),
            'NotifyClient': grpc.unary_unary_rpc_method_handler(
                    servicer.NotifyClient,
                    request_deserializer=market__pb2.NotifyClientRequest.FromString,
                    response_serializer=market__pb2.NotifyClientResponse.SerializeToString,
            ),
            'GetItemRatings': grpc.unary_unary_rpc_method_handler(
                    servicer.GetItemRatings,
                    request_deserializer=market__pb2.GetItemRatingsRequest.FromString,
                    response_serializer=market__pb2.GetItemRatingsResponse.SerializeToString,
            ),
            'NotifySeller': grpc.unary_unary_rpc_method_handler(
                    servicer.NotifySeller,
                    request_deserializer=market__pb2.NotifySellerRequest.FromString,
                    response_serializer=market__pb2.NotifySellerResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'Market', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Market(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def RegisterSeller(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Market/RegisterSeller',
            market__pb2.RegisterSellerRequest.SerializeToString,
            market__pb2.RegisterSellerResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SellItem(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Market/SellItem',
            market__pb2.SellItemRequest.SerializeToString,
            market__pb2.SellItemResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def UpdateItem(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Market/UpdateItem',
            market__pb2.UpdateItemRequest.SerializeToString,
            market__pb2.UpdateItemResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DeleteItem(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Market/DeleteItem',
            market__pb2.DeleteItemRequest.SerializeToString,
            market__pb2.DeleteItemResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DisplaySellerItems(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Market/DisplaySellerItems',
            market__pb2.DisplaySellerItemsRequest.SerializeToString,
            market__pb2.DisplaySellerItemsResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SearchItem(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Market/SearchItem',
            market__pb2.SearchItemRequest.SerializeToString,
            market__pb2.SearchItemResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def BuyItem(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Market/BuyItem',
            market__pb2.BuyItemRequest.SerializeToString,
            market__pb2.BuyItemResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def AddToWishList(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Market/AddToWishList',
            market__pb2.AddToWishListRequest.SerializeToString,
            market__pb2.AddToWishListResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def RateItem(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Market/RateItem',
            market__pb2.RateItemRequest.SerializeToString,
            market__pb2.RateItemResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def NotifyClient(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Market/NotifyClient',
            market__pb2.NotifyClientRequest.SerializeToString,
            market__pb2.NotifyClientResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetItemRatings(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Market/GetItemRatings',
            market__pb2.GetItemRatingsRequest.SerializeToString,
            market__pb2.GetItemRatingsResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def NotifySeller(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Market/NotifySeller',
            market__pb2.NotifySellerRequest.SerializeToString,
            market__pb2.NotifySellerResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)


class SellerStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.RecvNotification = channel.unary_unary(
                '/Seller/RecvNotification',
                request_serializer=market__pb2.RecvNotificationRequest.SerializeToString,
                response_deserializer=market__pb2.RecvNotificationResponse.FromString,
                )


class SellerServicer(object):
    """Missing associated documentation comment in .proto file."""

    def RecvNotification(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_SellerServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'RecvNotification': grpc.unary_unary_rpc_method_handler(
                    servicer.RecvNotification,
                    request_deserializer=market__pb2.RecvNotificationRequest.FromString,
                    response_serializer=market__pb2.RecvNotificationResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'Seller', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Seller(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def RecvNotification(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Seller/RecvNotification',
            market__pb2.RecvNotificationRequest.SerializeToString,
            market__pb2.RecvNotificationResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)


class ClientStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.RecieveNotification = channel.unary_unary(
                '/Client/RecieveNotification',
                request_serializer=market__pb2.RecieveNotificationRequest.SerializeToString,
                response_deserializer=market__pb2.RecieveNotificationResponse.FromString,
                )


class ClientServicer(object):
    """Missing associated documentation comment in .proto file."""

    def RecieveNotification(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ClientServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'RecieveNotification': grpc.unary_unary_rpc_method_handler(
                    servicer.RecieveNotification,
                    request_deserializer=market__pb2.RecieveNotificationRequest.FromString,
                    response_serializer=market__pb2.RecieveNotificationResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'Client', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Client(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def RecieveNotification(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Client/RecieveNotification',
            market__pb2.RecieveNotificationRequest.SerializeToString,
            market__pb2.RecieveNotificationResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
