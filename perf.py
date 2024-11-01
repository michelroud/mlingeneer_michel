from prefect import flow

PREFECT_API_URL="http://server:4200/api"
@flow
def my_flow( ) -> str:
    return "Hello, world!"

if __name__ == "__main__":
    print(my_flow())

