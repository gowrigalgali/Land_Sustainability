function loadGoogleMapsAPI() {
  const script = document.createElement('script');
  script.src = `https://maps.googleapis.com/maps/api/js?key=${CONFIG.GOOGLE_MAPS_API_KEY}&libraries=places,geometry&loading=async&callback=initMap`;
  script.async = true;
  script.defer = true;
  script.onerror = function() {
    console.error('Failed to load Google Maps API');
  };
  document.head.appendChild(script);
}

// Load Google Maps API when the page loads
window.onload = loadGoogleMapsAPI;