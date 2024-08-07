from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from gradio_client import Client
import httpx
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from PyPDF2 import PdfReader
from gtts import gTTS
import os

app = Flask(__name__)
CORS(app)


# Initialize Gradio Client for LLM with retry mechanism
@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3),
       retry=retry_if_exception_type(httpx.RequestError))
def initialize_gradio_client():
    return Client("osanseviero/mistral-super-fast")


client = initialize_gradio_client()


# Function to get response from LLM
@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3),
       retry=retry_if_exception_type(httpx.RequestError))
def get_response(prompt):
    result = client.predict(
        prompt=prompt,
        temperature=0.7,
        max_new_tokens=300,
        top_p=0.9,
        repetition_penalty=1.2,
        api_name="/chat"
    )
    response = result.strip()
    return response


@app.route('/generate_questions', methods=['POST'])
def generate_questions():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    resume_text = extract_text_from_pdf(file)

    prompt = (
        "Based on the following resume, generate a set of interview questions. "
        "The questions should be tailored to the candidate's experience and skills listed in the resume, "
        "and should be relevant to a typical job interview for a position in the related field.\n\n"
        "Resume:\n{resume_text}\n\nQuestions:"
    ).format(resume_text=resume_text)
    questions = get_response(prompt)
    question_list = [q.strip() for q in questions.split('\n') if q.strip()]

    return jsonify({'questions': question_list, 'resume_text': resume_text})


@app.route('/generate_follow_up', methods=['POST'])
def generate_follow_up():
    data = request.json

    if 'question' not in data or 'response' not in data:
        return jsonify({'error': 'Missing "question" or "response" key in request data'}), 400

    prompt = (
        "Based on the following response from the candidate to a specific question, generate one or two "
        "just ask the question based on the resume and don't say why you asked this specific question"
        "be a professional strict HR"
        "creative follow-up questions to explore the candidate's experience or skills further.\n\n"
        "Question: {question}\nCandidate's Response: {response}\nResume:\n{resume_text}\n\nFollow-Up Questions:"
    ).format(question=data['question'], response=data['response'], resume_text=data['resume_text'])

    response = get_response(prompt)
    follow_up_questions = [q.strip() for q in response.split('\n') if q.strip()]
    return jsonify({'follow_up_questions': follow_up_questions})


@app.route('/generate_feedback', methods=['POST'])
def generate_feedback():
    data = request.json
    interview_history = data['interview_history']

    prompt = (
        "Based on the following interview history, provide feedback on the candidate's performance. "
        "Highlight their strengths and areas for improvement to help them perform better in real-life interviews.\n\n"
        "Interview History:\n{interview_history}\n\nFeedback:"
    ).format(interview_history=interview_history)
    feedback = get_response(prompt)

    return jsonify({'feedback': feedback})


@app.route('/tts', methods=['POST'])
def generate_tts():
    text = request.json.get('text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    tts = gTTS(text, lang='en')
    tts.save("output.mp3")
    return send_file("output.mp3", mimetype='audio/mp3')


def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    resume_text = ""
    for page in pdf_reader.pages:
        resume_text += page.extract_text()
    return resume_text


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
