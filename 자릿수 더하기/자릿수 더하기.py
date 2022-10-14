def solution(n):
    answer = 0
    word = str(n)
    for i in range(len(word)):
        answer += int(word[i])
    return answer