import gradio as gr
from image_collage.image_process import ImageProcesser


image_processer = ImageProcesser(segments_dir="/home/rz60/codes/COMP646/COMP646_Project/segments_pool", 
                                     embed_path="./image_embeddings.csv")

def process_inputs(image, text_input=None, click_coords=None):
    message = "You uploaded an image."
    if text_input:
        message += f" Your text: '{text_input}'."
    if click_coords:
        message += f" Clicked at: {click_coords}."
    return image, message


def process_by_click_img(img, evt: gr.SelectData):
    
    coords = (evt.index[1], evt.index[0])
    print(coords)
    print(type(img))
    print(img.shape)
    img_processed = image_processer.click_process(img, coords)  # TODO: need a function to get a processed image
    return img_processed, coords


def process_by_text_prompt(img, text):
    prompt = text
    img_processed = img
    return img_processed


def replace_by_prompt(img, text):
    img_new = image_processer.replace(img, text)
    print(text)
    return img_new


def clear_image():
    image_processer.reset()


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="Input Image", show_download_button=True)
            input_img.clear(clear_image)
            
        # with gr.Column():
        #     output_img = gr.Image(label="Output Image")
    
    with gr.Row():
        
        with gr.Column():
            with gr.Tab("Click"):
                coord_text = gr.Text(label="Selected coordinates")
                input_img.select(process_by_click_img, [input_img], [input_img, coord_text])
                
            with gr.Tab("Text Prompt (step by step)"):
                select_prompt_text = gr.Text(label="Prompt to select")
                select_by_prompt_btn = gr.Button("Find the object")
                
                select_by_prompt_btn.click(fn=process_by_text_prompt, inputs=[input_img, select_prompt_text], outputs=[input_img])

        with gr.Column():
            replace_prompt_text = gr.Text(label="Prompt to replace")
            replace_by_prompt_btn = gr.Button("Run")
            
            replace_by_prompt_btn.click(replace_by_prompt, [input_img, replace_prompt_text], [input_img])
        
    
if __name__ == "__main__":
    demo.launch()