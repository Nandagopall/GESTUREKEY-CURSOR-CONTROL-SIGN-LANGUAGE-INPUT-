document.getElementById('mouseControl').addEventListener('click', function () {
    eel.set_mode('mouse')();  // Call Python function to switch to mouse control
});

document.getElementById('keyboardInput').addEventListener('click', function () {
    eel.set_mode('keyboard')();  // Call Python function to switch to keyboard input
});

document.getElementById('numberInput').addEventListener('click', function () {
    eel.set_mode('number')();  // Call Python function to switch to keyboard input
});