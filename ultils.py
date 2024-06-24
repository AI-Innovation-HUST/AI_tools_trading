import datetime


# Convert datetime to miliseconds
def datetime_to_ms(dt):
    dt = datetime.datetime.strptime(dt, '%Y-%m-%d %H:%M:%S.%f')
    return int(dt.timestamp() * 1000)

# Convert miliseconds to datetime
def ms_to_datetime(ms):
    return datetime.datetime.fromtimestamp(ms / 1000.0)

print(ms_to_datetime(1640969940000))

