import LOGO from '../assets/LOGO.svg';
import '../styles/navStyle.css';


function NavBar() {
    return (
        <div className='navContainer'>
            <img id= 'logo' src={LOGO} alt='MDvision logo'/>
            <div className='btnContainer'>
                {/* here is going to bee the rest of the buttons*/}
                <button className='navBtns'>Upload a Scan</button>
                <button className='navBtns'>Meet Maven</button>
                <button className='navBtns'>About us</button>
                <div>
                    <button className='navBtns' style={{color: '#1F59A2',  fontWeight: '500'}}>Login</button>|
                    <button className='navBtns' style={{color: '#d13f4e',  fontWeight: '500'}}>Signup</button>
                </div>
            </div>
        </div>
    );
}

export default NavBar;