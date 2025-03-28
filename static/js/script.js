document.getElementById('prediction-form').addEventListener('submit', function(event) {
    event.preventDefault();

    const city = document.getElementById('city').value;
    const property_type = document.getElementById('property_type').value;
    const surface = document.getElementById('surface').value;
    const bedroom = document.getElementById('bedroom').value;
    const bathroom = document.getElementById('bathroom').value;
    const principale = document.getElementById('principale').value || "Unknown";

    const data = {
        city: city,
        property_type: property_type,
        surface: surface,
        bedroom: bedroom,
        bathroom: bathroom,
        principale: principale
    };

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('predicted-price').innerText = data.predicted_price_MAD;
    })
    .catch(error => console.error('Error:', error));
});
