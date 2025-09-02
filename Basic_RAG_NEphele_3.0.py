# 
# ! pip install -U langchain-openai
# ! pip install langgraph
# ! pip install langgraph
# %pip install -qU langchain-community pdfminer.six
# ! pip install -U -q "google-genai>=1.16.0" # 1.16 is needed for multi-speaker audio



import getpass
import os,platform
from dotenv import load_dotenv
from google import genai
from google.genai import types
import wave
import base64
from IPython.display import Audio
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
import contextlib
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = "TEACHER_NEPHELE_3.0"
os.environ["USER_AGENT"] = "BasicRAGNephele/3.0"  # Set a custom user agent string


GEMINI_API_VOICE_KEY = os.getenv("GEMENI_APIKEY_VOICE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


prompt = hub.pull("rlm/rag-prompt")



# The code snippet `try:
#     llm = ChatOpenAI(model="gpt-4o")
# except Exception as e:
#     print(f"Exception : {e}")` is attempting to create an instance of the `ChatOpenAI` class with
# the model parameter set to "gpt-4o".
try:
    llm = ChatOpenAI(model="gpt-4o")
except Exception as e:
    print(f"Exception : {e}")



class RAG:
    def __init__(self, urls: list[str]):
        
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.vector_store = InMemoryVectorStore(self.embeddings)
        self.prompt = prompt  # <-- Fix: assign prompt to self.prompt
        
        try:
            loader = WebBaseLoader(
                web_paths=tuple(urls),
                bs_kwargs=dict(
                    parse_only=bs4.SoupStrainer(
                        class_=("post-content", "post-title", "post-header")
                    )
                ),
            )
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
            all_splits = text_splitter.split_documents(docs)
            self.vector_store.add_documents(documents=all_splits)
        except Exception as e:
            print(f"Error loading or processing documents: {e}")
            raise

    def retrieve(self, state: State):
        retrieved_docs = self.vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    def generate(self, state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = self.prompt.invoke({"question": state["question"], "context": docs_content})
        try:
            response = llm.invoke(messages)
            return {"answer": response.content}
        except Exception as e:
            print(f"Error during LLM generation: {e}")
            return {"answer": "Sorry, I encountered an error while generating the answer."}

    def build_graph(self):
        graph_builder = StateGraph(State).add_sequence([self.retrieve, self.generate])
        graph_builder.add_edge(START, "retrieve")
        graph = graph_builder.compile()
        return graph


class Audio_Generation:
    file_index = 0

    @staticmethod
    @contextlib.contextmanager
    def wave_file(filename, channels=1, rate=24000, sample_width=2):
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(rate)
            yield wf

    @classmethod
    # def play_audio_blob(cls, blob):
    #     cls.file_index += 1
    #     fname = f'audio_{cls.file_index}.wav'
    #     try:
    #         with cls.wave_file(fname) as wav:
    #             wav.writeframes(blob.data)
    #         return Audio(fname, autoplay=True)
    #     except Exception as e:
    #         print(f"Error writing audio file: {e}")
    #         return None

    
    def play_audio_blob(cls, blob):
        cls.file_index += 1
        fname = f'audio_{cls.file_index}.wav'
        try:
            with cls.wave_file(fname) as wav:
                wav.writeframes(blob.data)
            return fname  # return the filename instead of Audio
        except Exception as e:
            print(f"Error writing audio file: {e}")
            return None


def play_audio_from_text(text: str):
    if not text:
        print("No text provided for audio generation.")
        return None
    try:
        client = genai.Client(api_key=GEMINI_API_VOICE_KEY)

        MODEL_ID_VOICE = "gemini-2.5-flash-preview-tts"

        response = client.models.generate_content(
            model=MODEL_ID_VOICE,
            contents=text,
            config={"response_modalities": ["AUDIO"]},
        )

        blob = response.candidates[0].content.parts[0].inline_data
        return Audio_Generation.play_audio_blob(blob)
    
    except Exception as e:
        print(f"Error during audio generation: {e}")
        return None


if __name__ == '__main__':
    # --- Configuration ---
    DOCUMENT_URLS = ["https://lilianweng.github.io/posts/2023-03-15-prompt-engineering"]
    QUESTION = input("Enter your question: ")
    print('-'*100)
    try:
        # Initialize components
        rag = RAG(urls=DOCUMENT_URLS)
        graph = rag.build_graph()
        
        # Query execution
        res = graph.invoke({"question": QUESTION})
        answer = res.get("answer", "No answer found.")
        print(answer)
        
        # Generate and play audio
        if answer and answer != "Sorry, I encountered an error while generating the answer.":
            audio_output = play_audio_from_text(answer)
            if audio_output:
                print("Audio generated.")
                # In a script, the Audio object might not play automatically.
                # The returned object can be used in environments like Jupyter.
                # print(audio_output)
                if platform.system() == "Windows":
                    os.system(f"start {audio_output}")

    except Exception as e:
        print(f"An error occurred in the main execution block: {e}")










