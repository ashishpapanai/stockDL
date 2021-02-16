from flask import Flask
app = Flask("pyBhushan")

@app.route('/', methods=['GET'])

def ping():
    return "Pinging Model Application!"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)