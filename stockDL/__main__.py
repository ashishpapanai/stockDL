'''
Entry point of the library for website backends and CLI applications.
'''
from .main import Main


def main():
    ticker = input('Enter the ticker: ')
    main_obj = Main(ticker)


if __name__ == '__main__':
    main()
