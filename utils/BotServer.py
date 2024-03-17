import socket

class BotServer:
    def __init__(self, srv_port, listen_num):
        self.port = srv_port #listen port
        self.listen = listen_num
        self.mySock = None

    def create_socket(self):
        self.mySock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # IP and TCP 사용
        self.mySock.bind(('0.0.0.0', int(self.port))) # 0.0.0.0 --> Any vs '127.0.0.1 --> local 
        self.mySock.listen(int(self.listen))
        return self.mySock

    def ready_for_client(self):
        return self.mySock.accept()

    def get_sock(self):
        return self.mySock