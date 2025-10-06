function loadGoogleMapsAPI() {
  const script = document.createElement('script');
  script.src = `https://maps.googleapis.com/maps/api/js?key=${CONFIG.GOOGLE_MAPS_API_KEY}&libraries=places,geometry&callback=initMap`;
  script.async = true;
  script.defer = true;
  script.onerror = function() {
    console.error('Failed to load Google Maps API');
  };
  document.head.appendChild(script);
}

window.onload = loadGoogleMapsAPI;

function initMap() {
  const map = new google.maps.Map(document.getElementById('map'), {
    center: { lat: -34.397, lng: 150.644 },
    zoom: 8
  });
}