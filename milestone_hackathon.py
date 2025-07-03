import torch
import cv2
import numpy as np
from PIL import Image
from typing import List
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from prompt_generator import generate_prompt

# Configuration
VIDEO_PATH = "./2018-03-07.10-55-00.11-00-00.bus.G508.r13.avi"
NUM_FRAMES = 8  # Number of frames to extract from the video
OUTPUT_FILE = "2018-03-07.10-55-00.11-00-00.bus.G508.r13.txt"

# Load the Qwen model (using standard attention since FlashAttention2 is not installed)
print("Loading model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Configure the processor with optimized pixel settings
min_pixels = 256*28*28//3
max_pixels = 1280*28*28//3
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

def extract_frames(video_path, num_frames):
    """
    Extract evenly distributed frames from a video
    
    Args:
        video_path: Path to the video file
        num_frames: Number of frames to extract
    
    Returns:
        List of extracted frames as PIL Images
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")
    
    # Calculate frame indices to extract (evenly distributed)
    if total_frames <= num_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    
    # Extract the frames
    frames = []
    for idx in frame_indices:
        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        
        # Read the frame
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB (OpenCV uses BGR, PIL uses RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            pil_img = Image.fromarray(frame_rgb)
            frames.append(pil_img)
        else:
            print(f"Error: Could not read frame {idx}")
    
    # Release the video capture object
    cap.release()
    
    return frames

def analyze_video_frames(frames: List[Image.Image], backstory: str, last_batch: bool = False):
    """
    Analyze video frames using the Qwen model
    
    Args:
        frames: List of video frames as PIL Images
        backstory: Additional context for the analysis
        last_batch: Boolean indicating if this is the last batch
    
    Returns:
        Analysis text from the model
    """
    # Calculate frame positions based on batch and sampling rate
    num_frames = len(frames)
    start_frame = (batch_count - 1) * num_frames * 30
    end_frame = start_frame + (num_frames - 1) * 30
    
    # Generate appropriate prompt for this batch of frames
    analysis_prompt = generate_prompt(start_frame, end_frame)
    
    # Add system prompt
    system_prompt = "You are an expert surveillance video analyst. "
    if last_batch:
        system_prompt += "This is the last batch of frames from the video."
    else:
        system_prompt += "This is batch #{} of frames from the video.".format(batch_count)

    # Build messages with multiple frames
    contents = []
    
    # Add each frame to the contents
    for i, frame in enumerate(frames):
        contents.append({
            "type": "image",
            "image": frame,
        })
    
    # Add the text prompt at the end
    contents.append({
        "type": "text", 
        "text": analysis_prompt + ("\n\nPrevious analysis: " + backstory if backstory else ""),
    })
    
    messages = [{
        "role": "system",
        "content": system_prompt,
    },{
        "role": "user",
        "content": contents,
    }]
    
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)  # Ignore video_inputs as we're only using images
    
    print(f"Processing {len(image_inputs)} frames...")
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=None,  # Set videos to None since we're only processing images
        padding=True,
        return_tensors="pt",
    )
    
    # Move inputs to the same device as the model
    inputs = inputs.to(model.device)
    
    # Generate output
    print("Generating analysis...")
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text

def split_video_into_batches(video_path: str, batch_size: int = 10, frame_skip: int = 30):
    """
    Function that returns a generator yielding batches of frames from a video without loading everything into memory
    and the total number of expected batches.
    
    Samples 1 frame out of every frame_skip frames (default: 1 frame per second for 30fps video).
    
    Args:
        video_path: Path to the video file
        batch_size: Number of frames per batch
        frame_skip: Number of frames to skip between each sampled frame (e.g., 30 means take 1 frame per second in 30fps video)
        
    Returns:
        Tuple of (estimated_batch_count, frame_generator)
    """
    def frame_generator():
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        current_batch = []
        frame_count = 0
        batch_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                
                # Process only 1 frame per frame_skip frames
                if (frame_count - 1) % frame_skip == 0:
                    # Convert BGR (OpenCV) to RGB (PIL)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    
                    current_batch.append(pil_image)
                    
                    if len(current_batch) == batch_size:
                        batch_count += 1
                        yield current_batch
                        current_batch = []
            
            # Yield any remaining frames in the last batch
            if current_batch:
                batch_count += 1
                yield current_batch
        finally:
            # Release the video capture object
            cap.release()
    
    # Calculate an estimate of how many batches we'll have
    temp_cap = cv2.VideoCapture(video_path)
    if not temp_cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    total_frames = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    temp_cap.release()
    
    # Calculate expected frames after skipping
    expected_frames = total_frames // frame_skip + (1 if total_frames % frame_skip > 0 else 0)
    # Calculate expected batches
    expected_batches = expected_frames // batch_size + (1 if expected_frames % batch_size > 0 else 0)
    
    return expected_batches, frame_generator()
    
video_path = "./demo.avi"

# Process video in batches without loading everything into memory
print("Processing video in batches, sampling 1 frame per second (1 out of 30 frames)...")
total_batches, batch_generator = split_video_into_batches(video_path, batch_size=10, frame_skip=30)

# Process each batch individually
history = ""
batch_count = 0
for batch in batch_generator:
    batch_count += 1
    print(f"Processing batch {batch_count} with {len(batch)} frames")
    
    result = analyze_video_frames(batch, history, batch_count == total_batches)
    history += result[0]
    
    # Here you would do your processing with the batch
    # For example: result = analyze_video_frames(batch)
    
    # To avoid memory issues, don't store all batches/results
    # Instead, process and save results for each batch immediately
    
    # Free memory: explicitly delete the batch after processing
    del batch
    
print(f"Finished processing {batch_count} batches")
print(history)
# Save the output to a file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(history)

# Main execution
#print(f"Extracting {NUM_FRAMES} frames from {VIDEO_PATH}...")
#frames = extract_frames(VIDEO_PATH, NUM_FRAMES)
#print(f"Successfully extracted {len(frames)} frames.")
#
#if frames:
#    # Analyze the frames
#    analysis_results = analyze_video_frames(frames)
#
#    # Print the analysis results
#    for result in analysis_results:
#        print(result)
#
#    # Save the output to a file
#    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
#        for result in analysis_results:
#            f.write(result + "\n")
#
#    print(f"\nAnalysis saved to {OUTPUT_FILE}")

