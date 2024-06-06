import eel
from mousepgm import hand_mouse_control
from keyboard import aslkeyboard
from nkeyboard import num_keyboard

eel.init("web")  # Initialize Eel with the 'web' folder


@eel.expose  # Expose this function to JavaScript
def set_mode(mode):
    print("Received mode:", mode)
    if mode == "mouse":
        # Initialize gesture-based mouse control
        hand_mouse_control()
        print("Switched to Mouse Control")
    elif mode == "keyboard":
        aslkeyboard()
        # Initialize gesture-based keyboard input
        print("Switched to Keyboard Input")
    elif mode == "number":
        num_keyboard()
        print("Number Keyboard")


eel.start("index.html", size=(800, 600))  # Start Eel with the main HTML file
