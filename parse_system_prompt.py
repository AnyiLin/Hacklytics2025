import os
import base64
import json
import mimetypes

def encode_image_to_data_uri(image_path):
    """
    Encode a local image file to a Base64 data URI.
    If the file doesn't exist, return None.
    """
    if not os.path.exists(image_path):
        print(f"Warning: Image file '{image_path}' not found. Skipping.")
        return None
    with open(image_path, "rb") as img_file:
        data = img_file.read()
    encoded = base64.b64encode(data).decode("utf-8")
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "application/octet-stream"
    return f"data:{mime_type};base64,{encoded}"


def parse_markdown_to_messages(md_file_path):
    """
    Parse a Markdown file to produce a list of messages formatted for the ChatGPT API.
    
    Markdown structure assumptions:
      - Lines starting with "# " denote a new message and its role.
      - Lines after a header that do not start with "#" or "##" are part of a text block.
      - Lines starting with "## " denote an image element.
          • If the line text starts with "http", it is used directly as the URL.
          • Otherwise, it is treated as a local file path and encoded as a data URI.
          
    Each message is a dict with two keys:
      - "role": the role (e.g., "user", "assistant", "system")
      - "content": a list of message elements, each an object with:
          • For text: {"type": "text", "text": "some text"}
          • For images: {"type": "image_url", "image_url": {"url": "the-url-or-data-uri"}}
    """
    messages = []
    current_role = None
    current_parts = []  # List of message elements for the current message
    text_buffer = []    # To accumulate text lines

    def flush_text_buffer():
        nonlocal text_buffer, current_parts
        if text_buffer:
            # Combine the accumulated lines into one text block and append as a text element.
            text = "\n".join(text_buffer).strip()
            if text:
                current_parts.append({"type": "text", "text": text})
            text_buffer = []

    with open(md_file_path, "r", encoding="utf-8") as md_file:
        for line in md_file:
            stripped = line.strip()
            if stripped.startswith("# ") and not stripped.startswith("## "):
                # Starting a new message header.
                # Flush any accumulated text in the current message.
                flush_text_buffer()
                # If there is an active message, commit it.
                if current_role is not None:
                    messages.append({
                        "role": current_role,
                        "content": current_parts
                    })
                # Set new role (e.g., "user", "assistant", or "system")
                current_role = stripped[2:].strip()
                current_parts = []
            elif stripped.startswith("## "):
                # Flush any text accumulated so far before adding an image element.
                flush_text_buffer()
                # Get the image reference from the line.
                image_reference = stripped[3:].strip()
                # If it starts with http/https, assume it's a valid URL.
                if image_reference.lower().startswith("http"):
                    url_value = image_reference
                else:
                    # Otherwise, treat it as a local file path and encode it.
                    url_value = encode_image_to_data_uri(image_reference)
                    if url_value is None:
                        # Skip this image if it couldn't be encoded.
                        continue
                current_parts.append({
                    "type": "image_url",
                    "image_url": {"url": url_value}
                })
            elif stripped == "":
                # Blank line—if desired, you could add a newline.
                text_buffer.append("")  # preserve the break if needed
            else:
                # Regular text line.
                text_buffer.append(stripped)
        # End of file: flush any remaining text and commit the final message.
        flush_text_buffer()
        if current_role is not None:
            messages.append({
                "role": current_role,
                "content": current_parts
            })

    return messages


def write_json(output_file_path, data):
    """Write data to a JSON file with an indent for readability."""
    with open(output_file_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4)


def parse(md_file_path):
    messages = parse_markdown_to_messages(md_file_path)
    write_json("system_prompt.json", messages)
