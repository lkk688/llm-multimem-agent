from setuptools import setup, find_packages

setup(
    name="llm-multi-core",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "faiss-cpu",
        "sentence-transformers",
        "open-clip-torch",
        "torch",
        "Pillow",
        "fastapi[all]",
        "uvicorn",
        "gradio",
        "requests",
        "openai",
        "litellm"
    ],
    entry_points={
        "console_scripts": [
            "multimem=multimem.ui.gradio_app:launch_gradio"
        ]
    },
    author="Kaikai Liu",
    description="Multimodal Memory LLM Agent System",
    license="MIT"
)
