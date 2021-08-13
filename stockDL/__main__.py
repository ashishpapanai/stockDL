'''
Entry point of the library for website backends and CLI applications.
'''
from .main import Main


def main():
    ticker = input('Enter the ticker: ')
    saved = input('Do you want to use saved model: ')
    main_obj = Main(ticker, saved)


if __name__ == '__main__':
    main()
