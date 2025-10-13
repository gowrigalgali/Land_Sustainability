let map;
let polygon;
let coordinates = []; // Store polygon vertices
let markers = []; // Store markers for deselection


// Initialize the map
function initMap() {
  // Check if Google Maps API is loaded
  if (typeof google === 'undefined' || !google.maps) {
    console.error('Google Maps API not loaded');
    return;
  }

  map = new google.maps.Map(document.getElementById("map"), {
    center: { lat: 20.5937, lng: 78.9629 }, // Centered on India
    zoom: 5,
  });

  // Add search functionality
  const searchBox = new google.maps.places.SearchBox(
    document.getElementById("search-bar")
  );

  // Bias the SearchBox results towards the map's viewport
  map.addListener("bounds_changed", () => {
    searchBox.setBounds(map.getBounds());
  });

  // Handle search results
  searchBox.addListener("places_changed", () => {
    const places = searchBox.getPlaces();
    if (places.length === 0) return;

    const place = places[0];
    const location = place.geometry.location;

    addCoordinate(location);
    map.panTo(location);
    map.setZoom(15);
  });

  // Add event listener for map clicks
  map.addListener("click", (event) => {
    addCoordinate(event.latLng);
  });
}

// Add coordinates to the list and map
function addCoordinate(latLng) {
  const marker = new google.maps.Marker({
    position: latLng,
    map: map,
  });

  markers.push(marker);
  coordinates.push({ lat: latLng.lat(), lng: latLng.lng() });

  drawPolygon();

  // Allow deselecting a coordinate by clicking the marker
  marker.addListener("click", () => {
    removeCoordinate(marker);
  });
}

// Remove a coordinate and update the map
function removeCoordinate(marker) {
  const index = markers.indexOf(marker);
  if (index > -1) {
    markers[index].setMap(null);
    markers.splice(index, 1);
    coordinates.splice(index, 1);

    drawPolygon();
  }
}

// Draw the polygon on the map
function drawPolygon() {
  if (polygon) {
    polygon.setMap(null);
  }

  if (coordinates.length > 0) {
    polygon = new google.maps.Polygon({
      paths: coordinates,
      map: map,
      fillColor: "#00FF00",
      fillOpacity: 0.5,
      strokeColor: "#000000",
      strokeWeight: 2,
    });
    
    // Auto-calculate area if we have enough points
    if (coordinates.length >= 3) {
      try {
        const area = google.maps.geometry.spherical.computeArea(coordinates);
        const areaInKm2 = (area / 1000000).toFixed(2);
        const areaInHectares = (area / 10000).toFixed(2);
        
        let displayText;
        if (areaInKm2 < 1) {
          displayText = `Area: ${areaInHectares} hectares (${areaInKm2} km²)`;
        } else {
          displayText = `Area: ${areaInKm2} km²`;
        }
        
        document.getElementById("area-display").textContent = displayText;
      } catch (error) {
        console.error("Error auto-calculating area:", error);
      }
    } else {
      document.getElementById("area-display").textContent = "Area: -- km²";
    }
  }
}

// Function to handle coordinates saving to the backend
function handleCoordinatesWorkflow() {
  // Save coordinates to the server
  fetch("http://127.0.0.1:5001/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ coordinates }),
  })
    .then((response) => {
      if (!response.ok) throw new Error("Failed to save coordinates");
      return response.json();
    })
    .then((data) => {
      alert(data.message); // Notify user that saving was successful
    })
    .catch((error) => {
      console.error("Error:", error);
    });
}

// Clear all coordinates and markers
function clearMap() {
  if (polygon) polygon.setMap(null);
  markers.forEach((marker) => marker.setMap(null));
  markers = [];
  coordinates = [];
  document.getElementById("area-display").textContent = "Area: -- km²";
}

// Calculate area of the polygon
function calculateArea() {
  if (coordinates.length < 3) {
    alert("Please draw a polygon with at least 3 points to calculate area.");
    return;
  }

  try {
    // Use Google Maps geometry library to calculate area
    const area = google.maps.geometry.spherical.computeArea(coordinates);
    
    // Convert from square meters to square kilometers
    const areaInKm2 = (area / 1000000).toFixed(2);
    
    // Also show in hectares for smaller areas
    const areaInHectares = (area / 10000).toFixed(2);
    
    let displayText;
    if (areaInKm2 < 1) {
      displayText = `Area: ${areaInHectares} hectares (${areaInKm2} km²)`;
    } else {
      displayText = `Area: ${areaInKm2} km²`;
    }
    
    // Update the display
    document.getElementById("area-display").textContent = displayText;
    
    console.log(`Polygon area: ${areaInKm2} km² (${areaInHectares} hectares)`);
    
    // Show success message
    alert(`Area calculated successfully!\n${displayText}`);
    
  } catch (error) {
    console.error("Error calculating area:", error);
    alert("Error calculating area. Please make sure you have drawn a valid polygon.");
  }
}

// Wait for DOM to be ready before attaching event listeners
document.addEventListener('DOMContentLoaded', function() {
  // Attach event listener for the "Save Coordinates" button
  document
    .getElementById("search")
    .addEventListener("click", handleCoordinatesWorkflow);

  // Attach event listener for the "Calculate Area" button
  document
    .getElementById("calculate-area")
    .addEventListener("click", calculateArea);

  // Attach event listener for the "Clear Map" button
  document
    .getElementById("clear")
    .addEventListener("click", clearMap);

  // Attach event listener for the "Generate Report" button
  document
    .getElementById("report")
    .addEventListener("click", async function() {
      try {
        if (coordinates.length < 3) {
          alert("Please draw a polygon with at least 3 points before generating a report.");
          return;
        }

        const response = await fetch("http://127.0.0.1:5001/generate-report", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ coordinates }),
        });

        const contentType = response.headers.get('content-type') || '';

        if (!response.ok) {
          const errorBody = await response.text().catch(() => "");
          let message = "Failed to generate report";
          try {
            const parsed = JSON.parse(errorBody);
            if (parsed && parsed.message) message = parsed.message;
          } catch (_) {
            // Non-JSON error body; include status and snippet for debugging
            message = `${message} (status ${response.status}). ${errorBody?.slice(0, 200)}`;
          }
          console.error("Generate report failed", {
            status: response.status,
            contentType,
            body: errorBody,
          });
          throw new Error(message);
        }

        // If server didn't return a DOCX, surface the response as text
        if (!contentType.includes('application/vnd.openxmlformats-officedocument.wordprocessingml.document')) {
          const text = await response.text().catch(() => "");
          console.error("Unexpected content-type for report", { contentType, text });
          throw new Error(text || `Unexpected response type: ${contentType}`);
        }

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'report.docx';
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
      } catch (e) {
        console.error(e);
        alert("Error generating report. Please try again.");
      }
    });

  // Attach event listener for the "Generate Enhanced Report" button
  document
    .getElementById("enhanced-report")
    .addEventListener("click", async function() {
      // Show loading indicator
      const button = this;
      const originalText = button.textContent;
      button.textContent = "Generating Enhanced Report...";
      button.disabled = true;

      try {
        if (coordinates.length < 3) {
          alert("Please draw a polygon with at least 3 points before generating an enhanced report.");
          return;
        }

        const response = await fetch("http://127.0.0.1:5001/generate-enhanced-report", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ coordinates }),
        });

        const contentType = response.headers.get('content-type') || '';

        if (!response.ok) {
          const errorBody = await response.text().catch(() => "");
          let message = "Failed to generate enhanced report";
          try {
            const parsed = JSON.parse(errorBody);
            if (parsed && parsed.message) message = parsed.message;
          } catch (_) {
            message = `${message} (status ${response.status}). ${errorBody?.slice(0, 200)}`;
          }
          console.error("Generate enhanced report failed", {
            status: response.status,
            contentType,
            body: errorBody,
          });
          throw new Error(message);
        }

        // If server didn't return a DOCX, surface the response as text
        if (!contentType.includes('application/vnd.openxmlformats-officedocument.wordprocessingml.document')) {
          const text = await response.text().catch(() => "");
          console.error("Unexpected content-type for enhanced report", { contentType, text });
          throw new Error(text || `Unexpected response type: ${contentType}`);
        }

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'enhanced_report.docx';
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);

        alert("Enhanced report generated successfully! This report includes:\n" +
              "• Advanced ML models (ARIMA, Gradient Boosting, Ensemble)\n" +
              "• Environmental risk assessment\n" +
              "• Climate and air quality analysis\n" +
              "• Uncertainty quantification\n" +
              "• Comprehensive visualizations\n\n" +
              "Note: Processing time was optimized for faster generation.");

      } catch (e) {
        console.error(e);
        alert("Error generating enhanced report. Please try again.\n\nNote: Enhanced reports may take longer to generate due to additional data processing.");
      } finally {
        // Restore button state
        button.textContent = originalText;
        button.disabled = false;
      }
    });
});
