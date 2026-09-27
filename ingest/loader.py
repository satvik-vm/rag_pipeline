from langchain_community.document_loaders import (
    PyPDFium2Loader,
    WebBaseLoader,
)

def load_pdf(file_paths):
	data = []
	for file_path in file_paths:
		data.extend(PyPDFium2Loader(file_path).load())
	return data

def load_web(urls):
	data = WebBaseLoader(urls).load()
	return data
