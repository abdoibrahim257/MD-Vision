import LOGO from '../assets/LOGO.svg';
import humburger from '../assets/humburger.svg';
import close from '../assets/close.svg';
import { useState, useEffect } from 'react';
import {NavLink} from 'react-router-dom'; 
// import '../styles/topSection.css';



function NavBar({sticky = 0}) {

    const stickyBool = sticky === 1 ? true : false;

    const [fixed, setFixed] = useState(false);
    const [sideToggle, setSideToggle] = useState(false);


    if(!stickyBool){
        window.onscroll = () => {
            if(window.scrollY > 100){
                setFixed(true);
            }else{
                setFixed(false);
            }
        }
    }

    const toggleSideBar = () => {
        setSideToggle(!sideToggle);
    }

    return (
        <div className={fixed ? "navContainer content fixNav" : "navContainer content"}>
            <NavLink to="/">
                <img id= 'logo' src={LOGO} alt='MDvision logo'/>
            </NavLink>
            <div className='btnContainer'>
                {/* here is going to bee the rest of the buttons*/}
                <NavLink className = "navBtns hideOnMobile" to="/upload">Upload a Scan</NavLink>
                <NavLink className = "navBtns hideOnMobile" to="/maven">Meet Maven</NavLink>
                <NavLink className = "navBtns hideOnMobile" to="/about">About us</NavLink>
                {/* <NavLink className = "navBtns hideOnMobile" style={{color: '#d13f4e',  fontWeight: '600'}} to="/login">Login</NavLink> */}
                <img className = 'mobileShow' onClick={toggleSideBar} src={humburger} alt = 'menu button for mobile view'/>
            </div>

            <div className={sideToggle ? "sideBar" : "sideBarHidden"}>
                {/* here is going to bee the rest of the buttons*/}
                <img onClick ={toggleSideBar} src={close} alt = 'close button for mobile view'/>
                <NavLink className = "navBtns" to="/upload">Upload a Scan</NavLink>
                <NavLink className = "navBtns" to="/maven">Meet Maven</NavLink>
                <NavLink className = "navBtns" to="/about">About us</NavLink>
                {/* <NavLink className = "navBtns" style={{color: '#d13f4e',  fontWeight: '600'}} to="/login">Login</NavLink> */}
            </div>

        </div>
    );
}

export default NavBar;