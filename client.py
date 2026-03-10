import os
import grpc
import classifier_pb2
import classifier_pb2_grpc
from dotenv import load_dotenv
load_dotenv()

def classify(text: str) -> classifier_pb2.Classification:

    client_cert = os.getenv("CLIENT_CERT")
    server_addr = os.getenv("SERVER_ADDR")
    with open("ca.crt", "rb") as f:
        root_certs = f.read()

    credentials = grpc.ssl_channel_credentials(root_certificates=root_certs)
    
    with grpc.secure_channel(server_addr, credentials) as channel:
        stub = classifier_pb2_grpc.MBertClassifierStub(channel)
        return stub.Classify(classifier_pb2.ClassificationRequest(text=text))


if __name__ == "__main__":
    import sys

    text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Enter text: ")
    result = classify(text)
    print(f"Label:      {result.label}")
    print(f"Confidence: {result.confidence:.1%}")
