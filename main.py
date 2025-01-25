from dotenv import load_dotenv

load_dotenv()

from graph.graph import app

QUESTION = "How is the weather going to be in Ankara tomorrow?"

if __name__ == '__main__':
    print(app.invoke(input={"question": QUESTION}))