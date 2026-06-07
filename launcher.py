from app.launcher import *

if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000)
    except NameError:
        pass
