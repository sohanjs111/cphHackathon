def generate_prompt(start_frame, end_frame):
    base_prompt = (
        "You are an expert surveillance video analyst tasked with analyzing short video clips, each consisting of 5 consecutive frames. "
        "The full video is split into batches of frames for incremental analysis, e.g., frames 0-5, 6-10, 11-15, and so on. "
        "For the current clip covering frames {} to {}, please carefully observe the visual content.\n\n"
        "Your tasks for this clip:\n"
        "- Provide a detailed description of the scene, including people, objects, actions, and any relevant details.\n"
        "- Classify the clip as either 'Normal' or 'Anomalous' based on typical behavior and context.\n"
        "- Justify your classification with clear, evidence-based reasoning, avoiding unsupported assumptions.\n"
        "- Keep in mind the trade-off between using prior knowledge and avoiding hallucinations.\n\n"
        "Please consider typical activities and potential anomalies relevant to this environment.\n\n"
        "Respond with the following format:\n\n"
        "[Start frame - End frame]\n"
        "Scene Description:\n[Your detailed description here]\n\n"
        "Anomaly Classification:\n[Normal / Anomalous]\n\n"
        "Explanation:\n[Your reasoning here]\n\n"
        "Accurate and focused analysis is essential for reliable anomaly detection."
    ).format(start_frame, end_frame)
    
    anomaly_examples = "\n\nPlease analyze the scene based on general security and safety concerns relevant to the environment. At the very end, make a general assessment of the entire video: \nOverall assessment: (normal or abnormal)"
    
    return base_prompt + anomaly_examples
