from fastapi import APIRouter

router = APIRouter(prefix="/alerts")

@router.get("")
def get_alerts(lat: float, lng: float):
    return [
        {
            "location": "경기도 여주시",
            "type": "진딧물",
            "image": "https://example.com/sample.jpg",
            "tip": "초기 진딧물은 비눗물로 제거 가능"
        },
        {
            "location": "충남 논산시",
            "type": "잎마름병",
            "image": "https://example.com/sample2.jpg",
            "tip": "통풍 확보와 병든 잎 제거"
        }
    ]