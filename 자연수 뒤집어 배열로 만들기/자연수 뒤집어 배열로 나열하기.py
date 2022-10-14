def solution(n):
    answer = []
    for i in str(n):
        answer.append(i)
    answer = list(map(int,answer))
    answer.sort(reverse=True)
    return answer