import keyboard

while True:  # making a loop
    print(1)
    try:  # used try so that if user pressed other than the given key error will not be shown
        if keyboard.is_pressed('!'):  # if key 'q' is pressed
            print('You Pressed A Key!')
            break
    except:
        break  # if user pressed a key other than the given key the loop will break

print('end')