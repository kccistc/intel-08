uint32 ver
string src
uint32 seq
builtin_interfaces/Time ts  # 수신/발행 시각(또는 원본 ts 변환)

string type          # "collision", "fire", ...
string severity      # "low"|"medium"|"high"
float32 distance_m
string road
float64 lat
float64 lon

string suggest       # "slow_down"|"stop"|"reroute"|...

float32 ttl_s
