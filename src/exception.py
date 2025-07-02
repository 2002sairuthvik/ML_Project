# import sys
# import logging
# import logger as logger

# def error_message_detail(error,error_detail:sys):
#     _,_,exc_tb = error_detail.exc_info()
#     file_name = exc_tb.tb_frame.f_code.co_filename
#     line_number = exc_tb.tb_lineno
#     error_message = " Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
#         file_name,
#         line_number,
#         str(error)
#     )
#     return error_message
    
# class CustomException(Exception):
#     def __init__(self, error_message, error_detail:sys):
#         super().__init__(error_message)
#         self.error_message = error_message_detail(error_message, error_detail=error_detail)
    
#     def __str__(self):
#         return self.error_message


# if __name__ == "__main__":
#     try:
#         a = 1/0
#     except Exception as e:
#         logger.info("Division by zero error occurred")
#         raise CustomException(e,sys)
import sys
import logging
import logger  # This sets up logging config

def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = " Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name,
        line_number,
        str(error)
    )
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
        logging.error(self.error_message)  # Log the error automatically

    def __str__(self):
        return self.error_message

if __name__ == "__main__":
    try:
        a = 1/0
    except Exception as e:
        logging.info("Division by zero error occurred")  # <-- Use logging, not logger
        raise CustomException(e, sys)
