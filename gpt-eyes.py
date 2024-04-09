import cv2
import mediapipe as mp
import requests
import time
# from gtts import gTTS
import os

# Include your chat_with_gpt function here
api_key = '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'

def chat_with_gpt(prompt):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}],
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content'].strip()
    else:
        return "An error occurred: " + response.text


def main():
    # Initialize MediaPipe FaceMesh, Hands, and Pose
    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose

    face_mesh = mp_face_mesh.FaceMesh()
    hands = mp_hands.Hands()
    pose = mp_pose.Pose()

    video_path = "vegyn.mp4"
    cap = cv2.VideoCapture('GPT-Eyes/vegyn.mp4') #or video_path

    last_description_time = 0 
    description_interval = 10 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_results = face_mesh.process(rgb_frame)
        hand_results = hands.process(rgb_frame)
        pose_results = pose.process(rgb_frame)

        if face_results.multi_face_landmarks:
            for landmarks in face_results.multi_face_landmarks:
                draw_face_mesh(frame, landmarks, mp_face_mesh)

        if hand_results.multi_hand_landmarks:
            for landmarks in hand_results.multi_hand_landmarks:
                draw_hand_landmarks(frame, landmarks, mp_hands)

        if pose_results.pose_landmarks:
            draw_pose_landmarks(frame, pose_results.pose_landmarks, mp_pose)


        current_time = time.time()
        if current_time - last_description_time > description_interval:
            # Generate a prompt for GPT based on the current frame's analysis
            prompt = generate_gpt_prompt(face_results, hand_results, pose_results)
            description = chat_with_gpt(prompt + 'this is a description of a video that is being streamed, you are describing the video based on data you can understand from f{pose_results}, f{face_results}, f{draw_pose_landmarks}, f{draw_hand_landmarks}, f{draw_face_mesh}  and f{hand_results}, try to use the change in landmarks to depict what is happening in the scene, use pose changes to inform your ideas')
            print("GPT:", description)  # Example of using the description: printing it
                #    Create an instance of gTTS
            # tts = gTTS(text=description, lang='en')
            # tts.save("output.mp3")
            # os.system("afplay output.mp3")  #-----------------------------AUDIO PLAYER---------------------------------------
            last_description_time = current_time  # Update the last description time


        # Continue with MediaPipe processing and drawing as before
        cv2.imshow('Combined Tracking', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break


    cap.release()
    cv2.destroyAllWindows()

def draw_face_mesh(frame, landmarks, mp_face_mesh):
    h, w, _ = frame.shape
    mp_drawing = mp.solutions.drawing_utils

    # Convert normalized coordinates to pixel coordinates
    for landmark in landmarks.landmark:
        x, y = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    # Draw connections between facial landmarks
    mp_drawing.draw_landmarks(frame, landmarks, mp_face_mesh.FACEMESH_TESSELATION)

def draw_hand_landmarks(frame, landmarks, mp_hands):
    h, w, _ = frame.shape
    mp_drawing = mp.solutions.drawing_utils

    # Convert normalized coordinates to pixel coordinates
    for landmark in landmarks.landmark:
        x, y = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

    # Draw connections between hand landmarks
    mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

def draw_pose_landmarks(frame, landmarks, mp_pose):
    h, w, _ = frame.shape
    mp_drawing = mp.solutions.drawing_utils

    # Convert normalized coordinates to pixel coordinates
    for landmark in landmarks.landmark:
        x, y = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

    # Draw connections between body landmarks
    mp_drawing.draw_landmarks(frame, landmarks, mp_pose.POSE_CONNECTIONS)


def generate_gpt_prompt(face_results, hand_results, pose_results):
    # This function should generate a prompt based on the detections
    # For simplicity, let's say we're counting the number of faces, hands, and whether there's a pose detected.
    faces_count = len(face_results.multi_face_landmarks) if face_results.multi_face_landmarks else 0
    hands_count = len(hand_results.multi_hand_landmarks) if hand_results.multi_hand_landmarks else 0
    pose_detected = True if pose_results.pose_landmarks else False
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)


    prompt = f"Describe a scene with {faces_count} faces and {hands_count} hands visible. use whatever information you can get from {mp_face_mesh}, don't be over analytical use the change in position of the arms and head to determine the movement taking place "
    if pose_detected:
        prompt += " A person is posing."
 
    return prompt



if __name__ == "__main__":
    main()




 