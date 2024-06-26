import gradio as gr

from common import *

from Tools.ClassificationInference import classification_inference

EXIT_APP = False
log_file = "classification_inference.log"


# ----------------------------------------------------------------------------------------------------------------------
# Module
# ----------------------------------------------------------------------------------------------------------------------

def module_callback(images, points, model, class_map, patch_size, output_dir):
    """

    """
    console = sys.stdout
    sys.stdout = Logger(log_file)

    args = argparse.Namespace(
        images=images,
        points=points,
        model=model,
        class_map=class_map,
        patch_size=patch_size,
        output_dir=output_dir,
    )

    try:
        # Call the function
        gr.Info("Starting process...")
        classification_inference(args)
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
    logger = Logger(log_file)
    logger.reset_logs()

    with gr.Blocks(title="Predict 🤖️", analytics_enabled=False, theme=gr.themes.Soft(), js=js) as interface:
        # Title
        gr.Markdown("# Predict 🤖️")

        with gr.Group("Data"):
            #
            images = gr.Textbox(f"{DATA_DIR}", label="Selected Image Directory")
            dir_button = gr.Button("Browse Directory")
            dir_button.click(choose_directory, outputs=images, show_progress="hidden")

            points = gr.Textbox(label="Selected Points File")
            file_button = gr.Button("Browse Files")
            file_button.click(choose_file, outputs=points, show_progress="hidden")

            patch_size = gr.Number(112, label="Patch Size", precision=0)

        with gr.Group("Model"):
            #
            model = gr.Textbox(label="Selected Model File")
            file_button = gr.Button("Browse Files")
            file_button.click(choose_file, outputs=model, show_progress="hidden")

            class_map = gr.Textbox(label="Selected Class Map File")
            file_button = gr.Button("Browse Files")
            file_button.click(choose_file, outputs=class_map, show_progress="hidden")

        # Browse button
        output_dir = gr.Textbox(f"{DATA_DIR}", label="Selected Output Directory")
        dir_button = gr.Button("Browse Directory")
        dir_button.click(choose_directory, outputs=output_dir, show_progress="hidden")

        with gr.Row():
            # Run button (callback)
            run_button = gr.Button("Run")
            run = run_button.click(module_callback,
                                   [images,
                                    points,
                                    model,
                                    class_map,
                                    patch_size,
                                    output_dir])

            stop_button = gr.Button(value="Stop")
            stop = stop_button.click(exit_interface)

        with gr.Accordion("Console Logs"):
            # Add logs
            logs = gr.Code(label="", language="shell", interactive=False, container=True, lines=30)
            interface.load(logger.read_logs, None, logs, every=1)

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
    Logger(log_file).reset_logs()