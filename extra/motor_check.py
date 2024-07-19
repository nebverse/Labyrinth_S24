import pypot.dynamixel

ports = pypot.dynamixel.get_available_ports()
if not ports:
    raise IOError('no port found!')
print('ports found', ports)
for port in ports:
    print(port)
    dxl_io = pypot.dynamixel.DxlIO(port, baudrate=57600)
    print("scanning")
    found =  dxl_io.scan(range(60))
    print(found)
    dxl_io.close()
