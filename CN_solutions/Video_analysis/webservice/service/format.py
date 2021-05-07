def format_info(time_of_occur):
    for key in time_of_occur:
        duration = []
        time = time_of_occur[key][2]
        first = time[0]
        for i in range(1, len(time)):
            if i == len(time) - 1:
                if time[i] - time[i - 1] > 2:
                    duration.append([first, time[i - 1]])
                    duration.append([time[i]])
                elif first == time[i]:
                    duration.append([first])
                else:
                    duration.append([first, time[i]])
                break
            if time[i] - time[i - 1] <= 2:
                continue
            else:
                if first == time[i - 1]:
                    duration.append([first])
                else:
                    duration.append([first, time[i - 1]])
                first = time[i]
        time_of_occur[key][2] = duration
    return time_of_occur
