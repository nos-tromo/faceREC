import argparse
import datetime
import os

import cv2
import face_recognition
import urllib.request


def current_time():
    now = datetime.datetime.now().replace(microsecond=0)
    return now


def detect_faces(image):
    # Use face_recognition library for face detection
    face_locations = face_recognition.face_locations(image)
    faces = [(top, right, bottom, left) for top, right, bottom, left in face_locations]
    return faces


def expand_bbox(image, bbox, factor=1.5):
    # Expand the bounding box dimensions by a certain factor
    top, right, bottom, left = bbox
    width = right - left
    height = bottom - top
    expanded_top = max(0, top - int(height * (factor - 1) / 2))
    expanded_right = min(image.shape[1], right + int(width * (factor - 1) / 2))
    expanded_bottom = min(image.shape[0], bottom + int(height * (factor - 1) / 2))
    expanded_left = max(0, left - int(width * (factor - 1) / 2))
    return expanded_top, expanded_right, expanded_bottom, expanded_left


def are_faces_unique(
        new_faces,
        existing_faces,
        threshold=0.5,
        distance_threshold=100
):
    if len(existing_faces) == 0:
        return True

    for new_face in new_faces:
        x, y, w, h = new_face
        new_face_area = w * h
        new_face_centroid = (x + w // 2, y + h // 2)
        for existing_face in existing_faces:
            x_e, y_e, w_e, h_e = existing_face
            existing_face_centroid = (x_e + w_e // 2, y_e + h_e // 2)
            distance = ((new_face_centroid[0] - existing_face_centroid[0]) ** 2 +
                        (new_face_centroid[1] - existing_face_centroid[1]) ** 2) ** 0.5
            if distance < distance_threshold:
                intersection_area = max(0, min(x + w, x_e + w_e) - max(x, x_e)) * max(0, min(y + h, y_e + h_e) - max(y, y_e))
                union_area = new_face_area + (w_e * h_e) - intersection_area
                if union_area > 0:
                    overlap_ratio = intersection_area / union_area
                    if overlap_ratio > threshold:
                        return False
                else:
                    # If union_area is zero or negative, it's an edge case. For safety, consider the new face unique.
                    # This decision is arbitrary and should be aligned with the specific needs of your application.
                    # Log this event or handle accordingly if needed.
                    print("Encountered a zero or negative union area, which may indicate an issue with face bounding boxes.")
                    continue  # Consider the new face unique and continue checking other faces

    return True


def split_video_into_frames(
        video_path,
        output_folder,
        thumbnail_size=(200, 200),
        bbox_expansion_factor=1.5,
        frames_to_skip=30
):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open video file
    video_capture = cv2.VideoCapture(video_path)
    success, image = video_capture.read()
    count = 0
    frame_index = 0  # Initialize a frame index to count every frame read
    unique_faces = []

    # Read video frame by frame and save frames containing unique faces
    while success:
        # Process the frame only if frame_index modulo (frames_to_skip + 1) equals 0
        if frame_index % (frames_to_skip + 1) == 0:
            faces = detect_faces(image)
            if are_faces_unique(faces, unique_faces):
                for face in faces:
                    top, right, bottom, left = face
                    # Expand the bounding box to include more surrounding area
                    expanded_bbox = expand_bbox(image, face, factor=bbox_expansion_factor)
                    expanded_top, expanded_right, expanded_bottom, expanded_left = expanded_bbox
                    face_img = image[expanded_top:expanded_bottom, expanded_left:expanded_right]
                    # Resize the face image to a larger size
                    resized_face_img = cv2.resize(face_img, thumbnail_size)
                    cv2.imwrite(os.path.join(output_folder, f"frame_{frame_index}_face_{len(unique_faces)}.jpg"),
                                resized_face_img)

                    unique_faces.append(face)
            count += 1  # Increment count for each processed frame

        success, image = video_capture.read()
        frame_index += 1  # Increment frame index for every frame read, regardless of processing

    video_capture.release()


def main():
    print(f"Started: {current_time()}.")
    parser = argparse.ArgumentParser(description='Split a video into frames')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--file', help='Path to the video file')
    input_group.add_argument('--url', help='URL of the video file')
    parser.add_argument('--output_folder', default='frames', help='Folder to save frames (default: frames)')
    parser.add_argument('--thumbnail_size', default='200x200', help='Size of thumbnails in format "widthxheight" (default: 200x200)')
    parser.add_argument('--bbox_expansion_factor', type=float, default=1.5, help='Factor to expand the bounding box dimensions (default: 1.5)')
    parser.add_argument('--frames_to_skip', type=int, default=30, help='Number of frames to skip before processing the next frame (default: 30)')
    args = parser.parse_args()
    print(f"Frames to skip: {args.frames_to_skip}")

    output_folder = args.output_folder
    thumbnail_width, thumbnail_height = map(int, args.thumbnail_size.split('x'))
    thumbnail_size = (thumbnail_width, thumbnail_height)

    if args.file:
        video_path = args.file
    else:
        # If video_path is a URL, download the video
        video_file_name = os.path.basename(args.url)
        video_path = os.path.join('temp', video_file_name)
        urllib.request.urlretrieve(args.url, video_path)

    split_video_into_frames(video_path, output_folder, thumbnail_size, args.bbox_expansion_factor, args.frames_to_skip)
    print(f"Unique faces extracted and saved to '{output_folder}' folder.")
    print(f"Finished: {current_time()}.")


if __name__ == "__main__":
    main()
