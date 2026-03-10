import os
import grpc
import classifier_pb2
from infer import predict
import classifier_pb2_grpc
from concurrent import futures
from dotenv import load_dotenv

load_dotenv()


class ClassifierServicer(classifier_pb2_grpc.MBertClassifierServicer):
    def Classify(self, request, context):
        result = predict(request.text)
        return classifier_pb2.Classification(
            label=result["label"],
            confidence=result["confidence"],
        )


def serve():
    key = os.getenv("PRIVATE_KEY")
    cert_chain = os.getenv("SERVER_CERT")

    with open(key, "rb") as f:
        private_key = f.read()
    with open(cert_chain, "rb") as f:
        cert_chain = f.read()

    creds = grpc.ssl_server_credentials([(private_key, cert_chain)])
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))

    classifier_pb2_grpc.add_MBertClassifierServicer_to_server(ClassifierServicer(), server)
    server.add_secure_port("[::]:50051", creds)
    server.start()
    
    print("Server listening on port 50051")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
