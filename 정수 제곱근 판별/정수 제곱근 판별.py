def solution(n):
    answer = 0
    number = n ** 0.5
    if number == int(number):
        answer = (number+1)**2
    else :
        answer = -1
    return answer