import gradio as gr

from Toolbox.Pages.common import *

from Toolbox.Tools.Classification import classification
from Toolbox.Tools.Classification import get_classifier_losses
from Toolbox.Tools.Classification import get_classifier_encoders

EXIT_APP = False


# ----------------------------------------------------------------------------------------------------------------------
# Module
# ----------------------------------------------------------------------------------------------------------------------

def module_callback(patches, output_dir, encoder_name, loss_function, weighted_loss, augment_data, dropout_rate,
                    num_epochs, batch_size, learning_rate, tensorboard):
    """

    """
    console = sys.stdout
    sys.stdout = Logger(LOG_PATH)

    # Custom pre-processing
    patches = patches.split(" ")

    args = argparse.Namespace(
        patches=patches,
        output_dir=output_dir,
        encoder_name=encoder_name,
        loss_function=loss_function,
        weighted_loss=weighted_loss,
        augment_data=augment_data,
        dropout_rate=dropout_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        tensorboard=tensorboard,
    )

    try:
        # Call the function
        gr.Info("Starting process...")
        classification(args)
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

    with gr.Blocks(title="Train 👨‍💻", analytics_enabled=False, theme=gr.themes.Soft(), js=js) as interface:
        # Title
        gr.Markdown("Train 👨‍💻")

        # Browse button
        patches = gr.Textbox(label="Selected Patches File")
        file_button = gr.Button("Browse Files")
        file_button.click(choose_files, outputs=patches, show_progress="hidden")

        with gr.Row():
            encoder_name = gr.Dropdown(label="Encoder", multiselect=False, choices=get_classifier_encoders())

            freeze_encoder = gr.Dropdown(label="Freeze Encoder", multiselect=False, allow_custom_value=False,
                                         choices=[True, False])

        with gr.Row():
            optimizer = gr.Dropdown(label="Optimizer", multiselect=False, allow_custom_value=False,
                                    choices=[])

            learning_rate = gr.Slider(0.0001, label="Initial Learning Rate",
                                      minimum=0.00001, maximum=1, step=0.0001)

        with gr.Row():

            metrics = gr.Dropdown(label="Metrics", multiselect=True, choices=[])

            loss_function = gr.Dropdown(label="Loss Function", multiselect=False, choices=get_classifier_losses())

            weighted_loss = gr.Dropdown(label="Weighted Loss", multiselect=False, allow_custom_value=False,
                                        choices=[True, False])

        with gr.Row():
            augment_data = gr.Dropdown(label="Augment Data", multiselect=False, allow_custom_value=False,
                                       choices=[True, False])

            dropout_rate = gr.Slider(0, label="Dropout Rate", minimum=0, maximum=1, step=0.1)

        with gr.Row():
            num_epochs = gr.Number(25, label="Number of Epochs", precision=0)

            batch_size = gr.Number(128, label="Batch Size (Power of 2 Recommended)", precision=0)

            tensorboard = gr.Dropdown(label="Tensorboard", multiselect=False, allow_custom_value=False,
                                      choices=[True, False])

        output_dir = gr.Textbox(f"{DATA_DIR}", label="Selected Output Directory")
        dir_button = gr.Button("Browse Directory")
        dir_button.click(choose_directory, outputs=output_dir, show_progress="hidden")

        with gr.Row():
            # Run button (callback)
            run_button = gr.Button("Run")
            run = run_button.click(module_callback,
                                   [patches,
                                    output_dir,
                                    encoder_name,
                                    loss_function,
                                    weighted_loss,
                                    augment_data,
                                    dropout_rate,
                                    num_epochs,
                                    batch_size,
                                    learning_rate,
                                    tensorboard])

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