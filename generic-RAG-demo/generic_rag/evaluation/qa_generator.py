from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from generic_rag.evaluation.utils import format_docs


def build_qa_chain(chat_model, retriever, config: dict):
    """Build a QA chain with RAG prompt using required preloaded config."""

    # Get system prompt from config
    system_prompt = config["evaluation"]["prompts"]["qa_system"]

    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{question}")])
    return (
        {"context": lambda x: format_docs(retriever.invoke(x["question"])), "question": RunnablePassthrough()}
        | prompt
        | chat_model
        | StrOutputParser()
    )


def generate_answers(eval_df, qa_chain):
    eval_df["generated_answer"] = eval_df["question"].apply(lambda q: qa_chain.invoke({"question": q}))
    return eval_df
