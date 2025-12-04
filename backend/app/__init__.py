from .main import application 
# from fastapi.openapi.utils import get_openapi


# for route in application.routes:
#     print(route)
#     try:
#         get_openapi(
#             title="debug",
#             version="0.0.0",
#             routes=[route]
#         )
#     except Exception as e:
#         print("FAILED ROUTE:", route.path, route.name)
#         raise


__all__ = ["application"]
