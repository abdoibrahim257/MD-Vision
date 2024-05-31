import React from 'react'
import { FaSearch } from 'react-icons/fa'
import '../styles/SearchBar.css'

const SearchBar = ( { setResults } ) => {
    const [input, setInput] = React.useState('')

    const fetchData = (data) => {
        // fetch("https://jsonplaceholder.typicode.com/users")
        //     .then((response) => (response.json())) //geeting response and converting it to json
        //     .then((json) => {
        //         const results = json.filter((user) => {
        //             return data && user && user.name && user.name.toLowerCase().includes(data.toLowerCase());
        //         }); //this must be done on the backend side just for prototype
        //         console.log(results);
        //         // setResults(results);
        //     });
        fetch("https://bffe-35-237-74-64.ngrok-free.app/maven").then(response => response.json())
            .then(json => {
                const symptoms = json.fileList
                const results = symptoms.filter((symptom) => {
                    return data && (symptom.toLowerCase().includes(data.toLowerCase())) 
                })
                // console.log(results)
                setResults(results)
            });
    }

    const handleSearch = (data) => {
        setInput(data)
        fetchData(data)
    }

    
    return (
        <div className='input-wrapper' tabIndex="-1">
            <FaSearch id="search-icon" />
            <input type="text" id="search" placeholder="Search for your symptoms"
                value={input} onChange={(e) => handleSearch(e.target.value)} />
        </div>
    )
}

export default SearchBar