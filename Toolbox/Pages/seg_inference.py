import gradio as gr

from Toolbox.Pages.common import *

from Toolbox.Tools.SegInference import segmentation_inference

EXIT_APP = False


# ----------------------------------------------------------------------------------------------------------------------
# Module
# ----------------------------------------------------------------------------------------------------------------------

def module_callback(images, model, color_map, output_dir):
    """

    """
    console = sys.stdout
    sys.stdout = Logger(LOG_PATH)

    args = argparse.Namespace(
        images=images,
        model=model,
        color_map=color_map,
        output_dir=output_dir,
    )

    try:
        # Call the function
        gr.Info("Starting process...")
        segmentation_inference(args)
        print("\nDone.")
        gr.Info("Completed process!")
    except Exception as e:
        gr.Error("Could not complete process!")
        print(f"ERROR: {e}\n{traceback.format_exc()}")

    sys.stdout = console


# ----------------------------------------------------------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------------------------------------------------------
def exit_interface():
    """

    """
    global EXIT_APP
    EXIT_APP = True

    gr.Info("Please close the browser tab.")
    gr.Info("Stopped program successfully!")
    time.sleep(3)


def create_interface():
    """

    """
    Logger(LOG_PATH).reset_logs()

    with gr.Blocks(title="Predict 🤖️", analytics_enabled=False, theme=gr.themes.Soft(), js=js) as interface:
        # Title
        gr.Markdown("# Predict 🤖️")

        # Browse button
        images = gr.Textbox(label="Selected Images Directory")
        files_button = gr.Button("Browse Directory")
        files_button.click(choose_directory, outputs=images, show_progress="hidden")

        model = gr.Textbox(label="Selected Model File")
        files_button = gr.Button("Browse Files")
        files_button.click(choose_file, outputs=model, show_progress="hidden")

        color_map = gr.Textbox(label="Selected Color Map File")
        files_button = gr.Button("Browse Files")
        files_button.click(choose_file, outputs=color_map, show_progress="hidden")

        output_dir = gr.Textbox(f"{DATA_DIR}", label="Selected Output Directory")
        dir_button = gr.Button("Browse Directory")
        dir_button.click(choose_directory, outputs=output_dir, show_progress="hidden")

        with gr.Row():
            # Run button (callback)
            run_button = gr.Button("Run")
            run = run_button.click(module_callback,
                                   [images,
                                    model,
                                    color_map,
                                    output_dir])

            stop_button = gr.Button(value="Stop")
            stop = stop_button.click(exit_interface)

        with gr.Accordion("Console Logs"):
            # Add logs
            logs = gr.Code(label="", language="shell", interactive=False, container=True, lines=30)
            interface.load(read_logs, None, logs, every=1)

    interface.launch(prevent_thread_lock=True, server_port=get_port(), inbrowser=True, show_error=True)

    return interface


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------
interface = create_interface()

try:
    while True:
        time.sleep(0.5)
        if EXIT_APP:
            break
except:
    pass

finally:
    Logger(LOG_PATH).reset_logs()